"""
LoRA fine-tuning using PEFT library.

Key concept: LoRA (Low-Rank Adaptation) freezes the base model weights and injects
trainable rank-decomposition matrices into attention layers. ~0.3% of params vs 100%
in full fine-tuning, with comparable performance.

  W_adapted = W_frozen + (A @ B) * alpha/r
  where A ∈ R^(d×r), B ∈ R^(r×k), r << d (rank decomposition)

This script mirrors train.py but wraps the model with PEFT before the training loop.
Compare final metrics with models/full_finetuned/ to see the trade-off.
"""

import argparse
from pathlib import Path

import torch
import mlflow
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from model import load_model, load_tokenizer, LABEL2ID
from train import ReviewDataset, train_epoch, eval_epoch

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
LORA_DIR = Path(__file__).parent.parent / "models" / "lora_adapter"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_lora(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    patience: int = 3,
):
    device = get_device()
    print(f"Device: {device}")

    tokenizer = load_tokenizer()
    base_model = load_model()

    # Load from the fine-tuned checkpoint so the classifier head is already trained.
    # Starting from random head weights causes LoRA to fail to converge on small datasets.
    finetuned_dir = Path(__file__).parent.parent / "models" / "full_finetuned"
    if finetuned_dir.exists():
        print(f"Carregando checkpoint fine-tunado de: {finetuned_dir}")
        from model import load_finetuned
        base_model, _ = load_finetuned(str(finetuned_dir))
    else:
        print("Checkpoint fine-tunado não encontrado — usando base model (convergência mais lenta)")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # Apply LoRA to query and value projection matrices in every attention layer.
        # Key is left frozen — empirically, query+value is sufficient for classification.
        target_modules=["query", "value"],
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)

    print("\n--- Parâmetros treináveis vs total ---")
    model.print_trainable_parameters()
    trainable, total, pct = model.get_nb_trainable_parameters(), None, None
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"  Treináveis: {trainable_count:,} ({trainable_count/total_count*100:.2f}% do total)")

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
    mlflow.set_experiment("code-review-classifier-lora")
    with mlflow.start_run(run_name="lora"):
        mlflow.log_params({
            "model": "microsoft/codebert-base",
            "method": "LoRA",
            "lora_r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": "query,value",
            "trainable_params": trainable_count,
            "total_params": total_count,
            "trainable_pct": round(trainable_count / total_count * 100, 4),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        })

        best_val_loss = float("inf")
        patience_counter = 0
        LORA_DIR.mkdir(parents=True, exist_ok=True)

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
                # Save only the adapter weights (~3MB), not the full model (~500MB)
                model.save_pretrained(LORA_DIR)
                tokenizer.save_pretrained(LORA_DIR)
                print(f"  Adapter salvo (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("  Early stopping.")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        print(f"\nTreino LoRA concluído. Melhor val_loss: {best_val_loss:.4f}")
        print(f"Adapter salvo em: {LORA_DIR}")
        print(f"Tamanho aproximado: {sum((LORA_DIR / f).stat().st_size for f in ['adapter_model.safetensors'] if (LORA_DIR / f).exists()) / 1024:.0f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    train_lora(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.dropout,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )
