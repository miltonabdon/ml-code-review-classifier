"""
FastAPI serving endpoint — Code Review Classifier v0.2.0

Mudanças em relação à v0.1:
  - Eager loading no startup (não mais lazy)
  - POST /classify_batch com batching até 50 findings
  - GET /model_info — metadados do run MLflow que gerou o checkpoint
  - GET /health — estado detalhado: modelos, OOD, calibração, device
  - GET /version — versão da API e do checkpoint
  - Validação de input: 10–1000 chars por finding, batch máx 50
  - OOD detection via models/ood_thresholds.json (MSP threshold)
  - Temperature scaling via models/temperature.json
  - Retrocompatibilidade total com POST /classify
"""

import sys
import json
import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import load_finetuned, LABELS

logger = logging.getLogger("classifier_api")
logging.basicConfig(level=logging.INFO)

MODELS_DIR = Path(__file__).parent.parent / "models"
LORA_DIR = MODELS_DIR / "lora_adapter"
FULL_DIR = MODELS_DIR / "full_finetuned"
OOD_THRESHOLDS_PATH = MODELS_DIR / "ood_thresholds.json"
TEMPERATURE_PATH = MODELS_DIR / "temperature.json"

API_VERSION = "0.2.0"
MODEL_NAME = "codebert-base"

app = FastAPI(
    title="Code Review Classifier",
    description=(
        "Classifica findings de code review em: "
        "security, architecture, observability, style, false_positive"
    ),
    version=API_VERSION,
)

# ── Estado global ────────────────────────────────────────────────────────────

_state: Dict = {
    "models": {},       # "full" | "lora" -> (model, tokenizer, device)
    "ood": None,        # dict de thresholds por label, ou None
    "temperature": None,  # float, ou None
    "device": None,
}


# ── Pydantic models ──────────────────────────────────────────────────────────

class FindingRequest(BaseModel):
    finding: str = Field(..., min_length=10, max_length=1000)
    model: str = Field("lora", pattern="^(lora|full)$")

    @field_validator("finding")
    @classmethod
    def strip_finding(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError("finding deve ter ao menos 10 caracteres após strip")
        return stripped


class BatchRequest(BaseModel):
    findings: List[str] = Field(..., min_length=1, max_length=50)
    model: str = Field("lora", pattern="^(lora|full)$")

    @field_validator("findings")
    @classmethod
    def validate_findings(cls, findings: List[str]) -> List[str]:
        if len(findings) > 50:
            raise ValueError("batch máximo de 50 findings")
        result = []
        for i, f in enumerate(findings):
            stripped = f.strip()
            if len(stripped) < 10:
                raise ValueError(f"findings[{i}]: mínimo 10 caracteres")
            if len(stripped) > 1000:
                raise ValueError(f"findings[{i}]: máximo 1000 caracteres")
            result.append(stripped)
        return result


class ClassificationResult(BaseModel):
    cls: str
    confidence: float
    all_scores: Dict[str, float]
    model_used: str
    ood: Optional[bool] = None


# ── Device helper ────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Inference helper (síncrono — executado via run_in_executor) ───────────────

def _run_inference(model, tokenizer, device, texts: List[str], temperature: Optional[float]) -> List[dict]:
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # OOD check uses raw logits (no temperature) so the threshold is on the
    # same scale as the calibration data in ood_thresholds.json.
    raw_probs = torch.softmax(logits, dim=-1)

    if temperature is not None and temperature > 0:
        calibrated_logits = logits / temperature
    else:
        calibrated_logits = logits

    probs = torch.softmax(calibrated_logits, dim=-1)
    results = []
    for i in range(probs.shape[0]):
        p = probs[i]
        p_raw = raw_probs[i]
        pred_id = int(p.argmax().item())
        results.append({
            "pred_id": pred_id,
            "max_prob": float(p[pred_id].item()),
            "raw_max_prob": float(p_raw.argmax() == pred_id and p_raw[pred_id].item() or p_raw.max().item()),
            "scores": {label: round(float(p[j].item()), 4) for j, label in enumerate(LABELS)},
        })
    return results


def _apply_ood(result: dict, ood_thresholds: Optional[dict]) -> Optional[bool]:
    if ood_thresholds is None:
        return None
    import math
    # MSP check on raw (uncalibrated) probabilities
    msp_threshold = ood_thresholds.get("msp_threshold", 0.5)
    msp_flag = result["raw_max_prob"] < msp_threshold
    # Entropy check on calibrated probabilities (more reliable than MSP alone)
    scores = list(result["scores"].values())
    entropy = -sum(p * math.log(p + 1e-9) for p in scores)
    entropy_threshold = ood_thresholds.get("entropy_threshold", 1.4)
    entropy_flag = entropy > entropy_threshold
    return msp_flag or entropy_flag


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    device = _get_device()
    _state["device"] = device
    logger.info(f"Device: {device}")

    # Full model
    if FULL_DIR.exists():
        try:
            model, tokenizer = load_finetuned(str(FULL_DIR))
            model.to(device)
            model.eval()
            _state["models"]["full"] = (model, tokenizer, device)
            logger.info("Full model carregado.")
        except Exception as e:
            logger.error(f"Falha ao carregar full model: {e}")
    else:
        logger.warning(f"Full model não encontrado em {FULL_DIR}")

    # LoRA model
    if LORA_DIR.exists():
        try:
            from peft import PeftModel
            if "full" in _state["models"]:
                base, tok = load_finetuned(str(FULL_DIR))
            else:
                from model import load_model, load_tokenizer
                base = load_model(pretrained=False)
                tok = load_tokenizer()
            lora_model = PeftModel.from_pretrained(base, str(LORA_DIR))
            lora_model.to(device)
            lora_model.eval()
            _state["models"]["lora"] = (lora_model, tok, device)
            logger.info("LoRA model carregado.")
        except Exception as e:
            logger.error(f"Falha ao carregar LoRA model: {e}")
    else:
        logger.warning(f"LoRA adapter não encontrado em {LORA_DIR}")

    # OOD thresholds
    if OOD_THRESHOLDS_PATH.exists():
        with open(OOD_THRESHOLDS_PATH, encoding="utf-8") as f:
            _state["ood"] = json.load(f)
        logger.info(f"OOD thresholds carregados: {_state['ood']}")

    # Temperature scaling
    if TEMPERATURE_PATH.exists():
        with open(TEMPERATURE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _state["temperature"] = float(data.get("temperature", 1.0))
        logger.info(f"Temperature scaling: T={_state['temperature']}")


# ── MLflow helper ────────────────────────────────────────────────────────────

def _get_mlflow_model_info() -> dict:
    mlflow_db = Path(__file__).parent.parent / "mlflow.db"
    if not mlflow_db.exists():
        return {"error": "mlflow.db não encontrado"}
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        if not experiments:
            return {"error": "nenhum experimento no mlflow.db"}

        # Pega o run mais recente do experimento principal
        exp_names = [e.name for e in experiments]
        all_runs = []
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                all_runs.append((exp.name, runs[0]))

        if not all_runs:
            return {"error": "nenhum run encontrado"}

        # Escolhe o run mais recente entre os experimentos
        all_runs.sort(key=lambda x: x[1].info.start_time, reverse=True)
        exp_name, latest_run = all_runs[0]

        return {
            "experiment": exp_name,
            "run_id": latest_run.info.run_id,
            "status": latest_run.info.status,
            "start_time": latest_run.info.start_time,
            "metrics": latest_run.data.metrics,
            "params": latest_run.data.params,
        }
    except Exception as e:
        return {"error": str(e)}


def _get_checkpoint_name() -> str:
    if FULL_DIR.exists():
        config_path = FULL_DIR / "config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("_name_or_path", str(FULL_DIR))
    return str(FULL_DIR)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            "full": "full" in _state["models"],
            "lora": "lora" in _state["models"],
        },
        "ood_enabled": _state["ood"] is not None,
        "calibration_enabled": _state["temperature"] is not None,
        "device": str(_state.get("device", _get_device())),
    }


@app.get("/version")
async def version():
    return {
        "api": API_VERSION,
        "model": MODEL_NAME,
        "checkpoint": _get_checkpoint_name(),
    }


@app.get("/model_info")
async def model_info():
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(None, _get_mlflow_model_info)
    return info


@app.post("/classify", response_model=ClassificationResult)
async def classify(request: FindingRequest):
    model_key = request.model
    if model_key not in _state["models"]:
        avail = list(_state["models"].keys())
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modelo '{model_key}' não está carregado. "
                f"Disponíveis: {avail}. "
                f"Verifique se o checkpoint existe em {MODELS_DIR}."
            ),
        )

    model, tokenizer, device = _state["models"][model_key]
    temperature = _state["temperature"]
    ood_thresholds = _state["ood"]

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        _run_inference,
        model,
        tokenizer,
        device,
        [request.finding],
        temperature,
    )
    r = results[0]
    ood_flag = _apply_ood(r, ood_thresholds)

    model_label = f"codebert+{model_key}-finetuned"
    return ClassificationResult(
        cls=LABELS[r["pred_id"]],
        confidence=round(r["max_prob"], 4),
        all_scores=r["scores"],
        model_used=model_label,
        ood=ood_flag,
    )


@app.post("/classify_batch", response_model=List[ClassificationResult])
async def classify_batch(request: BatchRequest):
    model_key = request.model
    if model_key not in _state["models"]:
        avail = list(_state["models"].keys())
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modelo '{model_key}' não está carregado. "
                f"Disponíveis: {avail}. "
                f"Verifique se o checkpoint existe em {MODELS_DIR}."
            ),
        )

    model, tokenizer, device = _state["models"][model_key]
    temperature = _state["temperature"]
    ood_thresholds = _state["ood"]

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        _run_inference,
        model,
        tokenizer,
        device,
        request.findings,
        temperature,
    )

    model_label = f"codebert+{model_key}-finetuned"
    output = []
    for r in results:
        ood_flag = _apply_ood(r, ood_thresholds)
        output.append(ClassificationResult(
            cls=LABELS[r["pred_id"]],
            confidence=round(r["max_prob"], 4),
            all_scores=r["scores"],
            model_used=model_label,
            ood=ood_flag,
        ))
    return output


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("classifier_api:app", host="0.0.0.0", port=8000, reload=True)
