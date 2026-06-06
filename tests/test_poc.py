"""
15 testes de comportamento para a POC ML Code Review Classifier.

Não são testes unitários triviais — testam invariantes do sistema:
predições esperadas, calibração, OOD, baseline e API.
"""

import json
import math
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
FULL_DIR      = ROOT / "models" / "full_finetuned"
LORA_DIR      = ROOT / "models" / "lora_adapter"
TEMP_PATH     = ROOT / "models" / "temperature.json"
OOD_PATH      = ROOT / "models" / "ood_thresholds.json"
BASELINE_PATH = ROOT / "models" / "baseline_results.json"

needs_checkpoint = pytest.mark.skipif(
    not FULL_DIR.exists(), reason="Checkpoint não encontrado — rode python src/train.py"
)
needs_lora = pytest.mark.skipif(
    not LORA_DIR.exists(), reason="LoRA adapter não encontrado — rode python src/lora_train.py"
)
needs_calibration = pytest.mark.skipif(
    not TEMP_PATH.exists(), reason="Calibração não encontrada — rode python src/calibration.py"
)
needs_ood = pytest.mark.skipif(
    not OOD_PATH.exists(), reason="OOD thresholds não encontrados — rode python src/ood_detection.py"
)
needs_baseline = pytest.mark.skipif(
    not BASELINE_PATH.exists(), reason="Baseline não encontrado — rode python src/baseline.py"
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _predict(model, tokenizer, text: str) -> tuple[str, list[float]]:
    import torch
    from model import LABELS
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=256, padding="max_length")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred = LABELS[int(max(range(len(probs)), key=lambda i: probs[i]))]
    return pred, probs


# ════════════════════════════════════════════════════════════════════════════
# MODELO E INFERÊNCIA (5 testes)
# ════════════════════════════════════════════════════════════════════════════

@needs_checkpoint
def test_model_predicts_security_for_sql_injection(full_model_and_tokenizer):
    model, tok = full_model_and_tokenizer
    pred, _ = _predict(model, tok, "SQL query built with string concatenation — injection risk")
    assert pred == "security", f"Esperado 'security', obtido '{pred}'"


@needs_checkpoint
def test_model_predicts_observability_for_missing_logging(full_model_and_tokenizer):
    model, tok = full_model_and_tokenizer
    pred, _ = _predict(model, tok, "No logging on exception path — impossible to diagnose in production")
    assert pred == "observability", f"Esperado 'observability', obtido '{pred}'"


@needs_checkpoint
def test_model_confidence_is_between_0_and_1(full_model_and_tokenizer):
    model, tok = full_model_and_tokenizer
    _, probs = _predict(model, tok, "Variable name x is ambiguous — use descriptive name")
    assert abs(sum(probs) - 1.0) < 1e-4, f"Probs não somam 1: {sum(probs)}"
    assert all(0.0 <= p <= 1.0 for p in probs), f"Prob fora de [0,1]: {probs}"


@needs_checkpoint
@needs_lora
def test_lora_f1_not_worse_than_full_minus_threshold(full_model_and_tokenizer, lora_model_and_tokenizer):
    import json as _json
    from pathlib import Path
    from sklearn.metrics import f1_score
    import torch
    from model import LABELS

    test_examples = [_json.loads(l) for l in open(ROOT / "data" / "splits" / "test.jsonl")]

    def get_f1(model, tok):
        preds, labels = [], []
        for ex in test_examples:
            pred, _ = _predict(model, tok, ex["text"])
            preds.append(LABELS.index(pred))
            labels.append(LABELS.index(ex["label"]))
        return f1_score(labels, preds, average="macro")

    full_model, full_tok = full_model_and_tokenizer
    lora_model, lora_tok = lora_model_and_tokenizer

    f1_full = get_f1(full_model, full_tok)
    f1_lora = get_f1(lora_model, lora_tok)

    assert f1_lora >= f1_full - 0.05, (
        f"LoRA F1={f1_lora:.4f} é mais de 5pp inferior ao full FT F1={f1_full:.4f}"
    )


@needs_checkpoint
def test_model_returns_correct_class_per_example(full_model_and_tokenizer):
    model, tok = full_model_and_tokenizer
    cases = [
        ("SQL query built with string concatenation — injection risk", "security"),
        ("No logging on exception path — impossible to diagnose",       "observability"),
        ("Variable name x is ambiguous — use user_count",               "style"),
        ("This pattern is intentional — retry handles failures by design", "false_positive"),
    ]
    correct = 0
    for text, expected in cases:
        pred, _ = _predict(model, tok, text)
        if pred == expected:
            correct += 1
    # Pelo menos 3 de 4 devem estar corretos
    assert correct >= 3, f"Apenas {correct}/4 predições corretas"


# ════════════════════════════════════════════════════════════════════════════
# CALIBRAÇÃO (3 testes)
# ════════════════════════════════════════════════════════════════════════════

@needs_calibration
def test_temperature_json_exists_and_valid():
    data = json.loads(TEMP_PATH.read_text())
    assert "temperature" in data, "Campo 'temperature' ausente"
    T = float(data["temperature"])
    assert 0 < T < 10, f"Temperatura fora do intervalo (0, 10): T={T}"


@needs_calibration
def test_calibration_temperature_is_positive():
    data = json.loads(TEMP_PATH.read_text())
    T = float(data["temperature"])
    assert T > 0, f"Temperatura deve ser positiva, obtido T={T}"


@needs_calibration
def test_ece_decreases_after_calibration():
    data = json.loads(TEMP_PATH.read_text())
    ece_before = data.get("ece_before", 1.0)
    ece_after  = data.get("ece_after",  1.0)
    assert ece_after < ece_before, (
        f"ECE deveria diminuir após calibração: before={ece_before:.4f}, after={ece_after:.4f}"
    )


# ════════════════════════════════════════════════════════════════════════════
# OOD DETECTION (3 testes)
# ════════════════════════════════════════════════════════════════════════════

def test_entropy_is_maximum_for_uniform_distribution():
    from ood_detection import compute_entropy
    uniform = [0.2, 0.2, 0.2, 0.2, 0.2]
    H = compute_entropy(uniform)
    H_max = math.log(5)  # entropia máxima para 5 classes
    assert abs(H - H_max) < 0.01, f"Entropia uniforme esperada ~{H_max:.4f}, obtido {H:.4f}"


def test_entropy_is_zero_for_certain_prediction():
    from ood_detection import compute_entropy
    certain = [1.0, 0.0, 0.0, 0.0, 0.0]
    H = compute_entropy(certain)
    assert H < 0.01, f"Entropia de predição certa deveria ser ~0, obtido {H:.4f}"


@needs_ood
def test_ood_thresholds_json_exists_and_valid():
    data = json.loads(OOD_PATH.read_text())
    assert "msp_threshold" in data, "Campo 'msp_threshold' ausente"
    assert "entropy_threshold" in data, "Campo 'entropy_threshold' ausente"
    assert 0 < data["msp_threshold"] < 1, f"msp_threshold fora de (0,1): {data['msp_threshold']}"
    assert data["entropy_threshold"] > 0, f"entropy_threshold deve ser positivo"


# ════════════════════════════════════════════════════════════════════════════
# BASELINE (2 testes)
# ════════════════════════════════════════════════════════════════════════════

@needs_baseline
def test_baseline_f1_above_random():
    data = json.loads(BASELINE_PATH.read_text())
    f1 = data.get("f1_macro") or data.get("test_f1_macro", 0)
    assert f1 > 0.3, f"F1 do baseline ({f1:.4f}) deveria ser > 0.3 (random = 0.2 para 5 classes)"


@needs_baseline
def test_baseline_results_json_has_required_fields():
    data = json.loads(BASELINE_PATH.read_text())
    # Aceitar tanto f1_macro quanto test_f1_macro
    assert "f1_macro" in data or "test_f1_macro" in data, \
        "Campo 'f1_macro' ou 'test_f1_macro' ausente em baseline_results.json"
    assert "comparison" in data, "Campo 'comparison' ausente em baseline_results.json"
    cmp = data["comparison"]
    assert "delta" in cmp, "Campo 'delta' ausente em comparison"
    assert "verdict" in cmp, "Campo 'verdict' ausente em comparison"


# ════════════════════════════════════════════════════════════════════════════
# API E SERVING (2 testes)
# ════════════════════════════════════════════════════════════════════════════

@needs_checkpoint
def test_api_classify_returns_valid_schema(api_client):
    r = api_client.post("/classify", json={
        "finding": "SQL query built with string concatenation — injection risk",
        "model": "lora",
    })
    assert r.status_code == 200, f"Status inesperado: {r.status_code} — {r.text}"
    body = r.json()
    for field in ["cls", "confidence", "all_scores", "model_used"]:
        assert field in body, f"Campo '{field}' ausente na resposta"
    assert body["cls"] in ["security", "architecture", "observability", "style", "false_positive"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_api_rejects_short_input(api_client):
    r = api_client.post("/classify", json={"finding": "fix", "model": "lora"})
    assert r.status_code == 422, f"Input curto deveria retornar 422, obtido {r.status_code}"
