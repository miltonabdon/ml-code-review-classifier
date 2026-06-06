"""
Learning curves: F1 vs tamanho do dataset para CodeBERT e baseline TF-IDF.

Responde:
  - Quantos exemplos são necessários para atingir F1=0.90?
  - O modelo está saturado ou mais dados ajudariam?
  - O CodeBERT supera o baseline em todos os tamanhos de dataset?

Cada fração parte do modelo base pré-treinado (não do checkpoint fine-tunado)
para medir o efeito real do dataset, não do ponto de partida.
"""

import sys
import json
import random
import tempfile
from pathlib import Path

import torch
import mlflow
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from model import load_model, load_tokenizer, LABEL2ID, LABELS
from train import ReviewDataset, train_epoch, eval_epoch, get_device

ROOT = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"


def _subsample(dataset, fraction: float, seed: int) -> Subset:
    rng = random.Random(seed)
    n = max(1, int(len(dataset) * fraction))
    indices = rng.sample(range(len(dataset)), n)
    return Subset(dataset, indices)


def run_learning_curves(
    fractions: list[float] | None = None,
    n_epochs: int = 3,
    batch_size: int = 8,
) -> list[dict]:
    if fractions is None:
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    device = get_device()
    tokenizer = load_tokenizer()

    train_ds = ReviewDataset(SPLITS_DIR / "train.jsonl", tokenizer)
    val_ds   = ReviewDataset(SPLITS_DIR / "val.jsonl",   tokenizer)
    test_ds  = ReviewDataset(SPLITS_DIR / "test.jsonl",  tokenizer)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("learning-curves")

    results = []
    for fraction in fractions:
        seed = int(fraction * 100)
        subset = _subsample(train_ds, fraction, seed)
        n_train = len(subset)
        print(f"\nFração {fraction:.0%} — {n_train} exemplos de treino")

        # Partir sempre do base model pré-treinado (não do checkpoint)
        model = load_model(pretrained=True)
        model.to(device)

        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(loader) * n_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * 0.1)),
            num_training_steps=total_steps,
        )

        train_loss_final = 0.0
        with mlflow.start_run(run_name=f"frac_{fraction:.2f}"):
            mlflow.log_params({"fraction": fraction, "n_train": n_train, "epochs": n_epochs})
            for epoch in range(1, n_epochs + 1):
                train_loss, _ = train_epoch(model, loader, optimizer, scheduler, device)
                val_loss, val_acc = eval_epoch(model, val_loader, device)
                print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
                train_loss_final = train_loss

            # F1 no val set
            val_f1 = _compute_f1(model, val_loader, device)
            # F1 no test set
            test_f1 = _compute_f1(model, test_loader, device)

            mlflow.log_metrics({"val_f1": val_f1, "test_f1": test_f1})

        result = {
            "fraction": fraction,
            "n_train": n_train,
            "val_f1": round(val_f1, 4),
            "test_f1": round(test_f1, 4),
            "train_loss_final": round(train_loss_final, 4),
        }
        results.append(result)
        print(f"  val_f1={val_f1:.4f}  test_f1={test_f1:.4f}")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    return results


@torch.no_grad()
def _compute_f1(model, loader, device) -> float:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        all_preds.extend(out.logits.argmax(-1).cpu().tolist())
        all_labels.extend(batch["labels"].tolist())
    return f1_score(all_labels, all_preds, average="macro")


def run_baseline_learning_curves(fractions: list[float] | None = None) -> list[dict]:
    if fractions is None:
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    sys.path.insert(0, str(Path(__file__).parent))
    from baseline import train_baseline, evaluate_baseline

    import json as _json
    train_examples = [_json.loads(l) for l in open(SPLITS_DIR / "train.jsonl")]
    val_examples   = [_json.loads(l) for l in open(SPLITS_DIR / "val.jsonl")]
    test_examples  = [_json.loads(l) for l in open(SPLITS_DIR / "test.jsonl")]

    results = []
    for fraction in fractions:
        seed = int(fraction * 100)
        rng = random.Random(seed)
        n = max(1, int(len(train_examples) * fraction))
        subset = rng.sample(train_examples, n)
        print(f"\nBaseline fração {fraction:.0%} — {n} exemplos")

        # Salvar splits temporários
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.jsonl"
            val_path   = Path(tmp) / "val.jsonl"
            test_path  = Path(tmp) / "test.jsonl"
            train_path.write_text("\n".join(_json.dumps(e) for e in subset))
            val_path.write_text("\n".join(_json.dumps(e) for e in val_examples))
            test_path.write_text("\n".join(_json.dumps(e) for e in test_examples))

            pipeline = train_baseline(str(train_path), str(val_path))
            val_res  = evaluate_baseline(pipeline, str(val_path))
            test_res = evaluate_baseline(pipeline, str(test_path))

        result = {
            "fraction": fraction,
            "n_train": n,
            "val_f1": round(val_res["f1_macro"], 4),
            "test_f1": round(test_res["f1_macro"], 4),
        }
        results.append(result)
        print(f"  val_f1={val_res['f1_macro']:.4f}  test_f1={test_res['f1_macro']:.4f}")

    return results


def save_results(codebert_results: list[dict], baseline_results: list[dict]) -> None:
    path = MODELS_DIR / "learning_curves.json"
    data = {"codebert": codebert_results, "baseline": baseline_results}
    path.write_text(json.dumps(data, indent=2))
    print(f"\nResultados salvos em: {path}")


def plot_ascii(codebert: list[dict], baseline: list[dict]) -> None:
    print("\n=== Learning Curves ===")
    print(f"{'Fração':>8} {'N':>6} {'CodeBERT val_F1':>16} {'Baseline val_F1':>16}")
    print("-" * 52)
    bl_by_frac = {r["fraction"]: r for r in baseline}
    for r in codebert:
        bl = bl_by_frac.get(r["fraction"], {})
        bar_cb = "#" * int(r["val_f1"] * 20)
        bar_bl = "#" * int(bl.get("val_f1", 0) * 20)
        print(f"{r['fraction']:>7.0%} {r['n_train']:>6} {r['val_f1']:>7.4f} {bar_cb:<10} {bl.get('val_f1', 0):>7.4f} {bar_bl:<10}")


def print_summary(codebert: list[dict]) -> None:
    f1s = [r["val_f1"] for r in codebert]
    if len(f1s) < 2:
        return
    delta_last_two = f1s[-1] - f1s[-2]
    fraction_90 = next((r["fraction"] for r in codebert if r["val_f1"] >= 0.90), None)

    print("\n=== Summary ===")
    if fraction_90:
        print(f"F1=0.90 atingido com {fraction_90:.0%} do dataset")
    else:
        print(f"F1=0.90 não atingido (máximo: {max(f1s):.4f})")

    if delta_last_two < 0.01:
        print("→ Modelo SATURADO: ganho de 80%→100% dos dados foi < 1pp")
        print("  Mais dados ajudariam pouco. Foco em dados de melhor qualidade.")
    else:
        print(f"→ Modelo AINDA APRENDENDO: +{delta_last_two:.3f} F1 com 80%→100%")
        print("  Mais dados provavelmente ajudariam.")


if __name__ == "__main__":
    print("=== Baseline Learning Curves (rápido, sem GPU) ===")
    baseline_results = run_baseline_learning_curves()

    print("\n=== CodeBERT Learning Curves ===")
    codebert_results = run_learning_curves(n_epochs=3)

    save_results(codebert_results, baseline_results)
    plot_ascii(codebert_results, baseline_results)
    print_summary(codebert_results)
