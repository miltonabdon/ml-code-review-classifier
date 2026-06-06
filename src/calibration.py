"""
Temperature scaling calibration for CodeReviewClassifier.

Após treino, logits podem ser overconfident. Temperature scaling divide logits por T
aprendido no val set via minimização de NLL com LBFGS.
T > 1 → suaviza probabilidades (menos confiante)
T < 1 → sharpena probabilidades (mais confiante)
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from model import load_finetuned, LABELS
from train import ReviewDataset

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
MODELS_DIR = Path(__file__).parent.parent / "models"


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def fit(self, model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
        model.eval()
        self.to(device)

        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels.cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Optimise log(T) so T = exp(log_T) is always positive.
        log_T = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([log_T], lr=0.05, max_iter=200)
        nll_criterion = nn.CrossEntropyLoss()

        def eval_closure():
            optimizer.zero_grad()
            T = torch.exp(log_T).clamp(min=1e-4, max=10.0)
            scaled = logits / T
            loss = nll_criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

        T = float(torch.exp(log_T).clamp(min=1e-4, max=10.0).item())
        self.temperature.data = torch.tensor([T])
        return T

    @torch.no_grad()
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature.clamp(min=1e-6)
        return F.softmax(logits / T, dim=-1)


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Divide predições em n_bins por confiança máxima.
    ECE = sum_b (|B_b| / N) * |accuracy_b - confidence_b|
    """
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= low) & (confidences < high)
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)

        bin_size = mask.sum().item()
        if bin_size == 0:
            continue

        bin_acc = correct[mask].float().mean().item()
        bin_conf = confidences[mask].mean().item()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return ece


def reliability_diagram(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int]]:
    """
    Retorna (bin_confs, bin_accs, bin_counts) para plotar reliability diagram.
    bin_confs: confiança média de cada bin
    bin_accs: accuracy de cada bin
    bin_counts: número de amostras por bin
    """
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_confs: list[float] = []
    bin_accs: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= low) & (confidences < high)
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)

        bin_size = mask.sum().item()
        if bin_size == 0:
            bin_confs.append((low.item() + high.item()) / 2)
            bin_accs.append(0.0)
            bin_counts.append(0)
            continue

        bin_confs.append(confidences[mask].mean().item())
        bin_accs.append(correct[mask].float().mean().item())
        bin_counts.append(int(bin_size))

    return bin_confs, bin_accs, bin_counts


@torch.no_grad()
def _collect_logits_labels(
    model: nn.Module,
    tokenizer,
    split: str,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    split_path = SPLITS_DIR / f"{split}.jsonl"
    dataset = ReviewDataset(split_path, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.cpu())
        all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def calibrate_and_report(
    checkpoint_path: str,
    split: str = "val",
) -> tuple[float, tuple[list[float], list[float], list[int]]]:
    """
    Carrega modelo, roda temperature scaling no split, imprime ECE before/after.
    Salva T em models/temperature.json.
    Retorna (T, (bin_confs, bin_accs, bin_counts)).
    """
    device = _get_device()
    model, tokenizer = load_finetuned(checkpoint_path)
    model.to(device)
    model.eval()

    logits, labels = _collect_logits_labels(model, tokenizer, split, device)

    probs_before = F.softmax(logits, dim=-1)
    ece_before = compute_ece(probs_before, labels)

    dataset = ReviewDataset(SPLITS_DIR / f"{split}.jsonl", tokenizer)
    val_loader = DataLoader(dataset, batch_size=32)

    scaler = TemperatureScaler()
    T = scaler.fit(model, val_loader, device)

    probs_after = scaler.predict(logits)
    ece_after = compute_ece(probs_after, labels)

    print(f"Temperature T = {T:.4f}")
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after  calibration: {ece_after:.4f}")

    temp_path = MODELS_DIR / "temperature.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(json.dumps({"temperature": T, "ece_before": ece_before, "ece_after": ece_after}, indent=2))
    print(f"Temperatura salva em {temp_path}")

    diagram = reliability_diagram(probs_after, labels)
    return T, diagram


if __name__ == "__main__":
    checkpoint = str(Path(__file__).parent.parent / "models" / "full_finetuned")
    calibrate_and_report(checkpoint, split="val")
