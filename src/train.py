"""
Manual training loop for CodeReviewClassifier.

Intentionally avoids HuggingFace Trainer to expose the full training mechanics:
  - forward pass, loss computation, backward pass, optimizer step
  - train/val loss per epoch
  - MLflow experiment tracking
  - checkpoint on best val_loss with early stopping
"""

import json
import argparse
import os
from pathlib import Path

import torch
import mlflow
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from model import load_tokenizer, load_model, LABEL2ID, LABELS

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
MODELS_DIR = Path(__file__).parent.parent / "models" / "full_finetuned"


class ReviewDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = 256):
        self.examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.examples.append(row)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = LABEL2ID[ex["label"]]
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, scheduler, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="  val  ", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total


def train(
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    patience: int = 3,
    experiment_name: str = "code-review-classifier",
):
    device = get_device()
    print(f"Device: {device}")

    tokenizer = load_tokenizer()
    model = load_model()
    model.to(device)

    train_ds = ReviewDataset(SPLITS_DIR / "train.jsonl", tokenizer, max_length)
    val_ds = ReviewDataset(SPLITS_DIR / "val.jsonl", tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params({
            "model": "microsoft/codebert-base",
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "device": str(device),
        })

        best_val_loss = float("inf")
        patience_counter = 0
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss, val_acc = eval_epoch(model, val_loader, device)

            print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
            print(f"  val_loss={val_loss:.4f}    val_acc={val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model.save_pretrained(MODELS_DIR)
                tokenizer.save_pretrained(MODELS_DIR)
                print(f"  Checkpoint salvo (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Sem melhora ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print("  Early stopping.")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        print(f"\nTreino concluído. Melhor val_loss: {best_val_loss:.4f}")
        print(f"Modelo salvo em: {MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        patience=args.patience,
    )
