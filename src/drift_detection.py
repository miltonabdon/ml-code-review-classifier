"""
Drift detection for CodeReviewClassifier in production.

Two levels monitored:
  1. Prediction drift: distribution of predicted classes shifted (PSI)
  2. Confidence drift: KS test on max-softmax confidence distributions

PSI < 0.10   -> stable
PSI 0.10-0.25 -> moderate change (monitor)
PSI > 0.25   -> significant drift (alert)
"""

import sys
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent))
from model import load_finetuned, LABELS

ROOT = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
DEFAULT_REFERENCE_PATH = MODELS_DIR / "drift_reference.json"

EPSILON = 1e-6


@dataclass
class PredictionDistribution:
    label_counts: dict[str, int]
    label_fractions: dict[str, float]
    total: int
    mean_confidence: float
    mean_entropy: float
    timestamp: str
    # Raw confidences for KS test — persisted in JSON
    confidences: list[float] = None

    def __post_init__(self):
        if self.confidences is None:
            self.confidences = []


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _run_inference(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[list[int], list[float], list[float]]:
    """Returns (pred_ids, max_confidences, entropies) per text."""
    model.eval()
    model.to(device)

    preds, confidences, entropies = [], [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
        probs = F.softmax(logits, dim=-1)
        max_conf, pred_ids = probs.max(dim=-1)
        entropy = -(probs * (probs + EPSILON).log()).sum(dim=-1)

        preds.extend(pred_ids.cpu().tolist())
        confidences.extend(max_conf.cpu().tolist())
        entropies.extend(entropy.cpu().tolist())

    return preds, confidences, entropies


def _build_distribution(
    preds: list[int],
    confidences: list[float],
    entropies: list[float],
) -> PredictionDistribution:
    total = len(preds)
    counts = {label: 0 for label in LABELS}
    for p in preds:
        counts[LABELS[p]] += 1
    fractions = {label: counts[label] / total for label in LABELS}
    return PredictionDistribution(
        label_counts=counts,
        label_fractions=fractions,
        total=total,
        mean_confidence=float(np.mean(confidences)),
        mean_entropy=float(np.mean(entropies)),
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidences=confidences,
    )


def save_reference(dist: PredictionDistribution, path: str = str(DEFAULT_REFERENCE_PATH)) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(dist), indent=2))
    print(f"Referencia salva em: {p}")


def load_reference(path: str = str(DEFAULT_REFERENCE_PATH)) -> PredictionDistribution:
    data = json.loads(Path(path).read_text())
    return PredictionDistribution(**data)


def compute_reference_distribution(
    checkpoint_path: str,
    split: str = "train",
) -> PredictionDistribution:
    device = _get_device()
    model, tokenizer = load_finetuned(checkpoint_path)
    model.to(device)

    texts = [json.loads(line)["text"] for line in open(SPLITS_DIR / f"{split}.jsonl")]
    preds, confs, ents = _run_inference(model, tokenizer, texts, device)
    dist = _build_distribution(preds, confs, ents)

    save_reference(dist, str(DEFAULT_REFERENCE_PATH))
    print(f"Distribuicao de referencia calculada ({split}, {dist.total} exemplos):")
    for label, frac in dist.label_fractions.items():
        print(f"  {label:20s}: {frac:.1%}")
    return dist


def compute_current_distribution(
    checkpoint_path: str,
    texts: list[str],
) -> PredictionDistribution:
    device = _get_device()
    model, tokenizer = load_finetuned(checkpoint_path)
    model.to(device)
    preds, confs, ents = _run_inference(model, tokenizer, texts, device)
    return _build_distribution(preds, confs, ents)


def psi(p_ref: dict[str, float], p_curr: dict[str, float]) -> float:
    """
    Population Stability Index.
    PSI = sum((p_curr - p_ref) * ln(p_curr / p_ref))
    """
    score = 0.0
    for label in LABELS:
        ref = p_ref.get(label, 0.0) + EPSILON
        curr = p_curr.get(label, 0.0) + EPSILON
        score += (curr - ref) * math.log(curr / ref)
    return round(score, 6)


def ks_test_confidence(
    ref_confidences: list[float],
    curr_confidences: list[float],
) -> dict:
    stat, p_value = ks_2samp(ref_confidences, curr_confidences)
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": bool(p_value < 0.05),
    }


def detect_drift(
    reference: PredictionDistribution,
    current: PredictionDistribution,
) -> dict:
    psi_val = psi(reference.label_fractions, current.label_fractions)

    if psi_val < 0.10:
        psi_status = "stable"
    elif psi_val < 0.25:
        psi_status = "moderate"
    else:
        psi_status = "significant"

    ks = ks_test_confidence(reference.confidences, current.confidences)

    label_drift = {
        label: abs(
            current.label_fractions.get(label, 0.0)
            - reference.label_fractions.get(label, 0.0)
        )
        for label in LABELS
    }
    most_drifted = max(label_drift, key=label_drift.__getitem__)

    if psi_status == "significant" or (psi_status == "moderate" and ks["drift_detected"]):
        verdict = "alert"
    elif psi_status == "moderate" or ks["drift_detected"]:
        verdict = "monitor"
    else:
        verdict = "no_drift"

    summary = (
        f"PSI={psi_val:.3f} ({psi_status}) | "
        f"KS p={ks['p_value']:.4f} ({'drift' if ks['drift_detected'] else 'ok'}) | "
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


class DriftMonitor:
    def __init__(
        self,
        checkpoint_path: str,
        reference_path: str = str(DEFAULT_REFERENCE_PATH),
    ):
        self.checkpoint_path = checkpoint_path
        self.reference_path = reference_path
        self._reference: Optional[PredictionDistribution] = None

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
        all_examples = [json.loads(line) for line in open(SPLITS_DIR / "train.jsonl")]
        by_label = {label: [] for label in LABELS}
        for ex in all_examples:
            by_label[ex["label"]].append(ex["text"])

        if drift_type == "label_shift":
            # Flood with security findings to skew distribution away from reference
            security = by_label.get("security", [])
            false_pos = by_label.get("false_positive", [])[:2]
            texts = security * 5 + false_pos

        elif drift_type == "confidence_drop":
            # Interleave architecture/style findings — model is least confident at this boundary
            arch = by_label.get("architecture", [])
            style = by_label.get("style", [])
            texts = [t for pair in zip(arch, style) for t in pair][:20]
            if len(texts) < 10:
                texts = (arch + style) * 3

        elif drift_type == "domain_shift":
            # Non-code-review texts — model forced to classify out-of-domain inputs
            texts = [
                "Please send the quarterly report by end of day Friday",
                "Meeting rescheduled to 3pm, agenda: budget review Q2",
                "Lorem ipsum dolor sit amet consectetur adipiscing elit",
                "Pizza com abacaxi e uma combinacao controversa mas popular",
                "The quick brown fox jumps over the lazy dog near the river",
                "Quarterly earnings exceeded analyst expectations by 12 percent",
                "Reserve a window seat for the 9am flight to Sao Paulo",
                "The documentary explores the history of jazz in New Orleans",
                "Apply sunscreen 30 minutes before sun exposure for best results",
                "Water the plant twice a week and keep in indirect sunlight",
                "The new policy requires all employees to complete training by Q3",
                "Mix 200g of flour with 3 eggs and a pinch of salt then knead",
            ] * 3

        else:
            raise ValueError(
                f"drift_type desconhecido: '{drift_type}'. "
                "Esperado: 'label_shift', 'confidence_drop', 'domain_shift'"
            )

        result = self.check(texts)
        return texts, result


def _print_result(scenario: str, result: dict) -> None:
    psi_val = result["psi"]
    psi_status = result["psi_status"].upper()
    most_drifted = result["most_drifted_label"]
    drift_pct = result["label_drift"][most_drifted] * 100
    verdict = result["overall_verdict"].upper()

    print(f"\nCenario: {scenario}")
    print(f"PSI: {psi_val:.3f} -> {psi_status} DRIFT" if psi_status != "STABLE" else f"PSI: {psi_val:.3f} -> STABLE")
    print(f"Label mais desviado: {most_drifted} (+{drift_pct:.1f}%)")
    print(
        f"KS p-value: {result['ks_confidence']['p_value']:.4f} "
        f"({'drift detected' if result['ks_confidence']['drift_detected'] else 'no drift'})"
    )
    print(f"Veredicto: {verdict}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drift detection for CodeReviewClassifier")
    parser.add_argument(
        "--checkpoint",
        default=str(MODELS_DIR / "full_finetuned"),
        help="Path to the finetuned checkpoint",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint

    print("=" * 60)
    print("Calculando distribuicao de referencia no train set...")
    print("=" * 60)
    reference = compute_reference_distribution(checkpoint, split="train")
    print(f"\nTotal amostras: {reference.total}")
    print(f"Confianca media: {reference.mean_confidence:.4f}")
    print(f"Entropia media: {reference.mean_entropy:.4f}")

    monitor = DriftMonitor(checkpoint_path=checkpoint)
    monitor._reference = reference

    print("\n" + "=" * 60)
    print("Simulando 3 cenarios de drift...")
    print("=" * 60)

    for scenario in ["label_shift", "confidence_drop", "domain_shift"]:
        texts, result = monitor.simulate_drift(drift_type=scenario)
        _print_result(scenario, result)

    print("\n" + "=" * 60)
    print(f"Referencia salva em: {DEFAULT_REFERENCE_PATH}")
    print("=" * 60)
