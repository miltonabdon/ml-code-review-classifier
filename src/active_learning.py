"""
Active learning loop with uncertainty sampling over CodeBERT.

Selects the most uncertain examples from an unlabeled pool (highest entropy),
simulates human annotation via keyword matching, and re-trains incrementally
to measure F1 improvement per round.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from model import load_finetuned, LABEL2ID, LABELS, ID2LABEL

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"

# Copied from data/prepare_dataset.py to avoid cross-package import
LABEL_KEYWORDS = {
    "security": [
        "sql injection", "xss", "csrf", "authentication", "authorization",
        "password", "token", "secret", "credential", "sanitize", "escape",
        "vulnerability", "exploit", "privilege", "injection", "unsafe",
        "hardcoded", "plaintext", "encrypt", "hash", "tls", "ssl", "cert",
        "input validation", "deserialization", "path traversal", "rce",
    ],
    "architecture": [
        "coupling", "cohesion", "dependency", "interface", "abstraction",
        "single responsibility", "open closed", "solid", "design pattern",
        "separation of concerns", "modularity", "encapsulation", "inheritance",
        "composition", "refactor", "layer", "domain", "bounded context",
        "god class", "circular", "tight coupling", "responsibility",
    ],
    "observability": [
        "logging", "log", "tracing", "trace", "metric", "monitor",
        "instrumentation", "observable", "debug", "audit", "event",
        "correlation id", "structured log", "span", "telemetry", "alert",
        "dashboard", "health check", "error tracking",
    ],
    "style": [
        "naming", "variable name", "function name", "method name",
        "readability", "formatting", "indent", "whitespace", "comment",
        "documentation", "docstring", "typo", "consistency", "convention",
        "style", "clean code", "magic number", "magic string", "dead code",
        "unused", "redundant",
    ],
}


def _classify_text_keyword(text: str):
    text_lower = text.lower()
    scores = defaultdict(int)
    for label, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[label] += 1
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_score = ranked[0][1]
    top_labels = [l for l, s in ranked if s == best_score]
    return top_labels[0] if len(top_labels) == 1 else None


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


class _SimpleDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = LABEL2ID[ex["label"]]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_uncertainty(model, tokenizer, texts: list[str], device) -> list[float]:
    """Returns Shannon entropy H = -sum(p * log(p + 1e-9)) per text."""
    model.eval()
    entropies = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
            entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
            entropies.append(entropy)

    return entropies


def select_uncertain(
    model,
    tokenizer,
    pool: list[dict],
    n: int = 10,
    device=None,
) -> list[dict]:
    """Returns top-n pool examples ranked by highest entropy."""
    if device is None:
        device = _get_device()

    texts = [ex["text"] for ex in pool]
    entropies = compute_uncertainty(model, tokenizer, texts, device)

    model.eval()
    results = []
    with torch.no_grad():
        for ex, entropy in zip(pool, entropies):
            enc = tokenizer(
                ex["text"],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
            top_idx = int(np.argmax(probs))
            results.append({
                "text": ex["text"],
                "entropy": float(entropy),
                "top1": ID2LABEL[top_idx],
                "top1_prob": float(probs[top_idx]),
            })

    results.sort(key=lambda x: x["entropy"], reverse=True)
    return results[:n]


def simulate_annotation(examples: list[dict]) -> list[dict]:
    """Auto-annotate via keyword matching. Drops examples with ambiguous/no match."""
    annotated = []
    for ex in examples:
        label = _classify_text_keyword(ex["text"])
        if label is None:
            continue
        annotated.append({**ex, "label": label})
    return annotated


def _evaluate_f1(model, tokenizer, test_examples: list[dict], device) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    preds, truths = [], []

    with torch.no_grad():
        for ex in test_examples:
            enc = tokenizer(
                ex["text"],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            pred_idx = int(outputs.logits.argmax(dim=-1).item())
            preds.append(ID2LABEL[pred_idx])
            truths.append(ex["label"])

    return f1_score(truths, preds, average="macro")


def _quick_finetune(model, tokenizer, train_examples: list[dict], device, epochs: int = 2, batch_size: int = 8):
    """In-place fine-tuning for `epochs` epochs at lr=5e-5. Does not save checkpoint."""
    dataset = _SimpleDataset(train_examples, tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()


def active_learning_cycle(
    checkpoint_path: str,
    n_rounds: int = 3,
    n_per_round: int = 10,
) -> list[dict]:
    device = _get_device()
    print(f"Device: {device}")

    model, tokenizer = load_finetuned(checkpoint_path)
    model = model.to(device)

    train_examples = _load_jsonl(str(SPLITS_DIR / "train.jsonl"))
    val_examples = _load_jsonl(str(SPLITS_DIR / "val.jsonl"))
    test_examples = _load_jsonl(str(SPLITS_DIR / "test.jsonl"))

    # Val set serves as the unlabeled pool (simulation)
    pool = [{"text": ex["text"]} for ex in val_examples]

    baseline_f1 = _evaluate_f1(model, tokenizer, test_examples, device)
    print(f"F1 baseline (pre-AL): {baseline_f1:.4f}")

    history = []
    prev_f1 = baseline_f1
    current_train = list(train_examples)

    for round_idx in range(1, n_rounds + 1):
        print(f"\n--- Round {round_idx}/{n_rounds} ---")

        if not pool:
            print("Pool esgotado.")
            break

        selected = select_uncertain(model, tokenizer, pool, n=n_per_round, device=device)
        annotated = simulate_annotation(selected)
        n_added = len(annotated)
        print(f"  Selecionados: {n_per_round} | Anotados com sucesso: {n_added}")

        # Remove selected from pool (match by text)
        selected_texts = {ex["text"] for ex in selected}
        pool = [ex for ex in pool if ex["text"] not in selected_texts]

        if n_added == 0:
            print("  Nenhum exemplo anotável — pulando re-treino.")
            history.append({
                "round": round_idx,
                "examples_added": 0,
                "f1": prev_f1,
                "delta": 0.0,
            })
            continue

        current_train = current_train + annotated
        _quick_finetune(model, tokenizer, current_train, device, epochs=2, batch_size=8)

        new_f1 = _evaluate_f1(model, tokenizer, test_examples, device)
        delta = new_f1 - prev_f1
        print(f"  F1 após round {round_idx}: {new_f1:.4f} (delta={delta:+.4f})")

        history.append({
            "round": round_idx,
            "examples_added": n_added,
            "f1": round(new_f1, 4),
            "delta": round(delta, 4),
        })
        prev_f1 = new_f1

    return history


if __name__ == "__main__":
    checkpoint = str(Path(__file__).parent.parent / "models" / "full_finetuned")

    print("=== Active Learning Cycle ===")
    history = active_learning_cycle(checkpoint, n_rounds=2, n_per_round=10)

    print("\n=== Histórico de Rounds ===")
    for entry in history:
        print(
            f"  Round {entry['round']}: "
            f"+{entry['examples_added']} exemplos | "
            f"F1={entry['f1']:.4f} | "
            f"delta={entry['delta']:+.4f}"
        )

    print("\n=== Exemplos mais incertos (round 1 simulation) ===")
    from model import load_finetuned as _lf
    device = _get_device()
    model, tokenizer = _lf(checkpoint)
    model = model.to(device)
    val_examples = _load_jsonl(str(SPLITS_DIR / "val.jsonl"))
    pool = [{"text": ex["text"]} for ex in val_examples]
    uncertain = select_uncertain(model, tokenizer, pool, n=5, device=device)
    for i, ex in enumerate(uncertain, 1):
        print(f"  {i}. entropy={ex['entropy']:.4f} top1={ex['top1']} ({ex['top1_prob']:.2%})")
        print(f"     \"{ex['text'][:100]}...\"" if len(ex["text"]) > 100 else f"     \"{ex['text']}\"")
