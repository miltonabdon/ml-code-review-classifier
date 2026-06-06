"""
Drift detection para monitoramento em produção.

Dois níveis:
  1. Prediction drift — distribuição de classes preditas mudou (PSI)
  2. Confidence drift — distribuição de confiança mudou (KS test)

PSI < 0.10  → estável
PSI 0.10–0.25 → mudança moderada (monitorar)
PSI > 0.25  → drift significativo (alertar)
"""

import sys
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import load_finetuned, LABELS

ROOT = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"

EPS = 1e-6


@dataclass
class PredictionDistribution:
    label_counts: dict
    label_fractions: dict
    confidences: list
    total: int
    mean_confidence: float
    mean_entropy: float
    timestamp: str


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _run_inference(model, tokenizer, texts: list[str], device) -> tuple[list[int], list[float], list[float]]:
    preds, confidences, entropies = [], [], []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=256, padding="max_length")
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        pred = int(np.argmax(probs))
        conf = max(probs)
        ent = -sum(p * math.log(p + EPS) for p in probs)
        preds.append(pred)
        confidences.append(conf)
        entropies.append(ent)
    return preds, confidences, entropies


def _build_distribution(preds, confidences, entropies) -> PredictionDistribution:
    counts = {l: 0 for l in LABELS}
    for p in preds:
        counts[LABELS[p]] += 1
    total = len(preds)
    fractions = {l: counts[l] / total for l in LABELS}
    return PredictionDistribution(
        label_counts=counts,
        label_fractions=fractions,
        confidences=confidences,
        total=total,
        mean_confidence=float(np.mean(confidences)),
        mean_entropy=float(np.mean(entropies)),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def compute_reference_distribution(checkpoint_path: str, split: str = "train") -> PredictionDistribution:
    device = _get_device()
    model, tokenizer = load_finetuned(checkpoint_path)
    model.to(device)
    model.eval()

    import json as _json
    texts = [_json.loads(l)["text"] for l in open(SPLITS_DIR / f"{split}.jsonl")]
    preds, confs, ents = _run_inference(model, tokenizer, texts, device)
    dist = _build_distribution(preds, confs, ents)

    save_reference(dist)
    print(f"Distribuição de referência calculada ({split}, {dist.total} exemplos):")
    for l, f in dist.label_fractions.items():
        print(f"  {l:20s}: {f:.1%}")
    return dist


def compute_current_distribution(checkpoint_path: str, texts: list[str]) -> PredictionDistribution:
    device = _get_device()
    model, tokenizer = load_finetuned(checkpoint_path)
    model.to(device)
    model.eval()
    preds, confs, ents = _run_inference(model, tokenizer, texts, device)
    return _build_distribution(preds, confs, ents)


def psi(p_ref: dict, p_curr: dict) -> float:
    total = 0.0
    for label in LABELS:
        r = p_ref.get(label, 0) + EPS
        c = p_curr.get(label, 0) + EPS
        total += (c - r) * math.log(c / r)
    return round(total, 6)


def ks_test_confidence(ref_confidences: list[float], curr_confidences: list[float]) -> dict:
    from scipy.stats import ks_2samp
    stat, p_value = ks_2samp(ref_confidences, curr_confidences)
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": bool(p_value < 0.05),
    }


def detect_drift(reference: PredictionDistribution, current: PredictionDistribution) -> dict:
    psi_val = psi(reference.label_fractions, current.label_fractions)

    if psi_val < 0.10:
        psi_status = "stable"
    elif psi_val < 0.25:
        psi_status = "moderate"
    else:
        psi_status = "significant"

    ks = ks_test_confidence(reference.confidences, current.confidences)

    label_drift = {
        l: round(abs(current.label_fractions.get(l, 0) - reference.label_fractions.get(l, 0)), 4)
        for l in LABELS
    }
    most_drifted = max(label_drift, key=label_drift.get)

    if psi_status == "significant" or (psi_status == "moderate" and ks["drift_detected"]):
        verdict = "alert"
    elif psi_status == "moderate" or ks["drift_detected"]:
        verdict = "monitor"
    else:
        verdict = "no_drift"

    summary = (
        f"PSI={psi_val:.3f} ({psi_status}) | "
        f"KS p={ks['p_value']:.3f} ({'drift' if ks['drift_detected'] else 'ok'}) | "
        f"Most drifted: {most_drifted} (+{label_drift[most_drifted]:.1%}) | "
        f"Verdict: {verdict.upper()}"
    )

    return {
        "psi": psi_val,
        "psi_status": psi_status,
        "ks_confidence": ks,
        "label_drift": label_drift,
        "most_drifted_label": most_drifted,
        "overall_verdict": verdict,
        "summary": summary,
    }


def save_reference(dist: PredictionDistribution, path: str | None = None) -> None:
    p = Path(path) if path else MODELS_DIR / "drift_reference.json"
    data = asdict(dist)
    p.write_text(json.dumps(data, indent=2))
    print(f"Referência salva em: {p}")


def load_reference(path: str | None = None) -> PredictionDistribution:
    p = Path(path) if path else MODELS_DIR / "drift_reference.json"
    data = json.loads(p.read_text())
    return PredictionDistribution(**data)


class DriftMonitor:
    def __init__(self, checkpoint_path: str, reference_path: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.reference_path = reference_path or str(MODELS_DIR / "drift_reference.json")
        self._reference: PredictionDistribution | None = None

    def load_reference(self) -> PredictionDistribution:
        self._reference = load_reference(self.reference_path)
        return self._reference

    def check(self, texts: list[str]) -> dict:
        if self._reference is None:
            self.load_reference()
        current = compute_current_distribution(self.checkpoint_path, texts)
        result = detect_drift(self._reference, current)
        result["n_examples"] = current.total
        result["current_distribution"] = current.label_fractions
        result["reference_distribution"] = self._reference.label_fractions
        return result

    def simulate_drift(self, drift_type: str = "label_shift") -> tuple[list[str], dict]:
        import json as _json

        all_examples = [_json.loads(l) for l in open(SPLITS_DIR / "train.jsonl")]
        by_label = {l: [] for l in LABELS}
        for ex in all_examples:
            by_label[ex["label"]].append(ex["text"])

        if drift_type == "label_shift":
            # Só envia findings de security — distribuição desloca para security
            texts = by_label.get("security", []) * 5
            texts += by_label.get("false_positive", [])[:2]
        elif drift_type == "confidence_drop":
            # Findings ambíguos na fronteira architecture/style
            arch = by_label.get("architecture", [])
            style = by_label.get("style", [])
            texts = arch[:5] + style[:5]
            # Repetir intercalado para criar distribuição mista
            texts = [t for pair in zip(arch, style) for t in pair][:20]
        elif drift_type == "domain_shift":
            # Textos fora do domínio de code review
            texts = [
                "Please send the quarterly report by end of day Friday",
                "Meeting rescheduled to 3pm, agenda: budget review",
                "SELECT * FROM users WHERE id = 1 ORDER BY created_at",
                "def connect(): conn = db.connect('localhost'); return conn",
                "Lorem ipsum dolor sit amet consectetur adipiscing elit",
                "Pizza com abacaxi é uma combinação controversa mas popular",
                "The quick brown fox jumps over the lazy dog near the river",
                "git push origin main --force --no-verify",
            ] * 3
        else:
            raise ValueError(f"drift_type desconhecido: {drift_type}")

        result = self.check(texts)
        return texts, result


def _print_result(scenario: str, result: dict) -> None:
    emoji = {"no_drift": "✅", "monitor": "⚠️", "alert": "🚨"}.get(result["overall_verdict"], "❓")
    print(f"\nCenário: {scenario}")
    print(f"  PSI: {result['psi']:.3f} → {result['psi_status'].upper()}")
    print(f"  KS p-value: {result['ks_confidence']['p_value']:.3f} "
          f"({'drift' if result['ks_confidence']['drift_detected'] else 'ok'})")
    print(f"  Label mais desviado: {result['most_drifted_label']} "
          f"(+{result['label_drift'][result['most_drifted_label']]:.1%})")
    print(f"  Veredicto: {emoji} {result['overall_verdict'].upper()}")


if __name__ == "__main__":
    checkpoint = str(MODELS_DIR / "full_finetuned")

    print("=== Calculando distribuição de referência (train set) ===")
    compute_reference_distribution(checkpoint, split="train")

    monitor = DriftMonitor(checkpoint)
    monitor.load_reference()

    print("\n=== Simulando cenários de drift ===")
    for scenario in ["label_shift", "confidence_drop", "domain_shift"]:
        texts, result = monitor.simulate_drift(scenario)
        _print_result(scenario, result)

    print("\n=== Cenário sem drift (val set normal) ===")
    import json as _json
    val_texts = [_json.loads(l)["text"] for l in open(SPLITS_DIR / "val.jsonl")]
    result = monitor.check(val_texts)
    _print_result("val set (baseline)", result)
