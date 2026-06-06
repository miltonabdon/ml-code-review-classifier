"""
Evaluation of trained model on test split.

Outputs:
  - F1 macro + weighted, precision/recall per class
  - Confusion matrix (console + saved as PNG)
  - Top-5 misclassified examples per category
"""

import json
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from model import LABELS, LABEL2ID, load_finetuned
from train import ReviewDataset

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
MODELS_DIR = Path(__file__).parent.parent / "models" / "full_finetuned"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def get_predictions(model, loader, device) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    for batch in tqdm(loader, desc="Avaliando"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = probs.argmax(dim=-1)
        confs = probs.max(dim=-1).values

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_confs.extend(confs.cpu().tolist())

    return all_preds, all_labels, all_confs


def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=LABELS,
        yticklabels=LABELS,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix salva em: {output_path}")


def show_errors(test_examples: list[dict], preds: list[int], top_n: int = 5):
    id2label = {i: l for i, l in enumerate(LABELS)}
    errors_by_true: dict[str, list] = {l: [] for l in LABELS}

    for i, (ex, pred) in enumerate(zip(test_examples, preds)):
        true_label = ex["label"]
        pred_label = id2label[pred]
        if true_label != pred_label:
            errors_by_true[true_label].append({
                "text": ex["text"][:100],
                "true": true_label,
                "pred": pred_label,
            })

    print("\n=== Top erros por categoria ===")
    for label in LABELS:
        errors = errors_by_true[label][:top_n]
        if not errors:
            print(f"\n[{label}] — sem erros")
            continue
        print(f"\n[{label}] — {len(errors_by_true[label])} erros total, mostrando {len(errors)}:")
        for e in errors:
            print(f"  pred={e['pred']}: {e['text']!r}")


def evaluate(checkpoint_path: Path = MODELS_DIR, batch_size: int = 32):
    device = get_device()
    print(f"Device: {device}")
    print(f"Carregando modelo de: {checkpoint_path}")

    model, tokenizer = load_finetuned(str(checkpoint_path))
    model.to(device)

    test_examples = []
    with open(SPLITS_DIR / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line))

    test_ds = ReviewDataset(SPLITS_DIR / "test.jsonl", tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    preds, labels, _ = get_predictions(model, test_loader, device)

    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, target_names=LABELS, digits=3))

    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    output_dir = checkpoint_path.parent.parent / "models"
    plot_confusion_matrix(labels, preds, output_dir / "confusion_matrix.png")
    show_errors(test_examples, preds)

    return {"f1_macro": f1_macro, "f1_weighted": f1_weighted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluate(Path(args.checkpoint), args.batch_size)
