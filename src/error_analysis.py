"""
Análise sistemática de erros do modelo além da confusion matrix.

Categorias de análise:
- by_class: erros por classe verdadeira
- confusion_pairs: pares (true, predicted) ordenados por frequência
- high_confidence_errors: erros com confiança > 0.7 (os mais perigosos)
- low_confidence_correct: acertos com confiança < 0.5 (acerto por sorte)
- avg_confidence_by_class: confiança média por classe verdadeira
- boundary_examples: fronteiras de decisão ambíguas (|p1 - p2| < 0.2)
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from model import load_finetuned, LABELS, ID2LABEL
from train import ReviewDataset

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
MODELS_DIR = Path(__file__).parent.parent / "models"


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def analyze_errors(
    checkpoint_path: str,
    split: str = "test",
) -> dict:
    device = _get_device()
    model, tokenizer = load_finetuned(checkpoint_path)
    model.to(device)
    model.eval()

    split_path = SPLITS_DIR / f"{split}.jsonl"
    dataset = ReviewDataset(split_path, tokenizer)
    loader = DataLoader(dataset, batch_size=32)

    # coleta raw texts para associar ao resultado
    import json
    raw_texts: list[str] = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            raw_texts.append(json.loads(line)["text"])

    all_probs: list[list[float]] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1).cpu()
        preds = probs.argmax(dim=-1)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    # --- by_class: erros agrupados por classe verdadeira ---
    by_class: dict[str, list[dict]] = defaultdict(list)
    for i, (probs, pred, true) in enumerate(zip(all_probs, all_preds, all_labels)):
        if pred != true:
            confidence = probs[pred]
            by_class[ID2LABEL[true]].append({
                "text": raw_texts[i],
                "predicted": ID2LABEL[pred],
                "confidence": round(confidence, 4),
                "correct": False,
            })

    # --- confusion_pairs: (true_label, pred_label) ordenados por freq ---
    pair_counter: Counter = Counter()
    for pred, true in zip(all_preds, all_labels):
        if pred != true:
            pair_counter[(ID2LABEL[true], ID2LABEL[pred])] += 1
    confusion_pairs = [
        {"true": k[0], "predicted": k[1], "count": v}
        for k, v in pair_counter.most_common()
    ]

    # --- high_confidence_errors: confiança > 0.7 e errado ---
    high_confidence_errors: list[dict] = []
    for i, (probs, pred, true) in enumerate(zip(all_probs, all_preds, all_labels)):
        confidence = probs[pred]
        if pred != true and confidence > 0.7:
            high_confidence_errors.append({
                "text": raw_texts[i],
                "true": ID2LABEL[true],
                "predicted": ID2LABEL[pred],
                "confidence": round(confidence, 4),
                "probs": {LABELS[j]: round(p, 4) for j, p in enumerate(probs)},
            })
    high_confidence_errors.sort(key=lambda x: x["confidence"], reverse=True)

    # --- low_confidence_correct: acertos com confiança < 0.5 ---
    low_confidence_correct: list[dict] = []
    for i, (probs, pred, true) in enumerate(zip(all_probs, all_preds, all_labels)):
        confidence = probs[pred]
        if pred == true and confidence < 0.5:
            low_confidence_correct.append({
                "text": raw_texts[i],
                "label": ID2LABEL[true],
                "confidence": round(confidence, 4),
                "probs": {LABELS[j]: round(p, 4) for j, p in enumerate(probs)},
            })
    low_confidence_correct.sort(key=lambda x: x["confidence"])

    # --- avg_confidence_by_class: confiança média por classe verdadeira ---
    class_conf_sum: dict[str, float] = defaultdict(float)
    class_conf_count: dict[str, int] = defaultdict(int)
    for probs, pred, true in zip(all_probs, all_preds, all_labels):
        confidence = probs[pred]
        class_conf_sum[ID2LABEL[true]] += confidence
        class_conf_count[ID2LABEL[true]] += 1

    avg_confidence_by_class = {
        label: round(class_conf_sum[label] / class_conf_count[label], 4)
        if class_conf_count[label] > 0 else 0.0
        for label in LABELS
    }

    # --- boundary_examples: |p1 - p2| < 0.2 (fronteira ambígua) ---
    boundary_examples: list[dict] = []
    for i, (probs, pred, true) in enumerate(zip(all_probs, all_preds, all_labels)):
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        top1_idx, top1_prob = sorted_probs[0]
        top2_idx, top2_prob = sorted_probs[1]
        gap = top1_prob - top2_prob
        if gap < 0.2:
            boundary_examples.append({
                "text": raw_texts[i],
                "true": ID2LABEL[true],
                "predicted": ID2LABEL[top1_idx],
                "correct": pred == true,
                "top1": {"label": ID2LABEL[top1_idx], "prob": round(top1_prob, 4)},
                "top2": {"label": ID2LABEL[top2_idx], "prob": round(top2_prob, 4)},
                "gap": round(gap, 4),
            })
    boundary_examples.sort(key=lambda x: x["gap"])

    # métricas globais
    total = len(all_labels)
    n_errors = sum(1 for p, t in zip(all_preds, all_labels) if p != t)
    accuracy = (total - n_errors) / total if total > 0 else 0.0

    return {
        "split": split,
        "total_examples": total,
        "total_errors": n_errors,
        "accuracy": round(accuracy, 4),
        "by_class": dict(by_class),
        "confusion_pairs": confusion_pairs,
        "high_confidence_errors": high_confidence_errors,
        "low_confidence_correct": low_confidence_correct,
        "avg_confidence_by_class": avg_confidence_by_class,
        "boundary_examples": boundary_examples,
    }


def print_report(analysis: dict) -> None:
    sep = "=" * 80
    thin = "-" * 80

    print(sep)
    print(f"ERROR ANALYSIS REPORT — split={analysis['split']}")
    print(sep)
    print(f"Total examples : {analysis['total_examples']}")
    print(f"Total errors   : {analysis['total_errors']}")
    print(f"Accuracy       : {analysis['accuracy']:.4f}")

    print(f"\n{'ERRORS BY CLASS':}")
    print(thin)
    for label in LABELS:
        errors = analysis["by_class"].get(label, [])
        print(f"  {label:20s}: {len(errors)} errors")
        for ex in errors[:3]:
            snippet = ex["text"][:70].replace("\n", " ")
            print(f"    → pred={ex['predicted']:20s} conf={ex['confidence']:.3f} | {snippet}")
        if len(errors) > 3:
            print(f"    ... ({len(errors) - 3} more)")

    print(f"\n{'CONFUSION PAIRS (top 10)':}")
    print(thin)
    for pair in analysis["confusion_pairs"][:10]:
        print(f"  {pair['true']:20s} → {pair['predicted']:20s} : {pair['count']}x")

    print(f"\n{'HIGH CONFIDENCE ERRORS (conf > 0.7)':} — {len(analysis['high_confidence_errors'])} total")
    print(thin)
    for ex in analysis["high_confidence_errors"][:5]:
        snippet = ex["text"][:65].replace("\n", " ")
        print(f"  conf={ex['confidence']:.3f} | true={ex['true']:15s} pred={ex['predicted']:15s}")
        print(f"  text: {snippet}")
        print(f"  probs: { {k: v for k, v in sorted(ex['probs'].items(), key=lambda x: -x[1])[:3]} }")
    if len(analysis["high_confidence_errors"]) > 5:
        print(f"  ... ({len(analysis['high_confidence_errors']) - 5} more)")

    print(f"\n{'LOW CONFIDENCE CORRECT (conf < 0.5)':} — {len(analysis['low_confidence_correct'])} total")
    print(thin)
    for ex in analysis["low_confidence_correct"][:5]:
        snippet = ex["text"][:65].replace("\n", " ")
        print(f"  conf={ex['confidence']:.3f} | label={ex['label']:15s}")
        print(f"  text: {snippet}")
    if len(analysis["low_confidence_correct"]) > 5:
        print(f"  ... ({len(analysis['low_confidence_correct']) - 5} more)")

    print(f"\n{'AVG CONFIDENCE BY CLASS (true label)':}")
    print(thin)
    for label, conf in analysis["avg_confidence_by_class"].items():
        bar_len = int(conf * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"  {label:20s}: {conf:.4f} |{bar}|")

    print(f"\n{'BOUNDARY EXAMPLES (|p1-p2| < 0.2, top 5 tightest)':}")
    print(thin)
    for ex in analysis["boundary_examples"][:5]:
        snippet = ex["text"][:60].replace("\n", " ")
        correct_mark = "OK" if ex["correct"] else "ERR"
        print(
            f"  [{correct_mark}] gap={ex['gap']:.3f} | "
            f"true={ex['true']:15s} "
            f"top1={ex['top1']['label']:15s}({ex['top1']['prob']:.3f}) "
            f"top2={ex['top2']['label']:15s}({ex['top2']['prob']:.3f})"
        )
        print(f"  text: {snippet}")
    if len(analysis["boundary_examples"]) > 5:
        print(f"  ... ({len(analysis['boundary_examples']) - 5} more)")

    print(sep)


if __name__ == "__main__":
    checkpoint = str(Path(__file__).parent.parent / "models" / "full_finetuned")
    analysis = analyze_errors(checkpoint, split="test")
    print_report(analysis)
