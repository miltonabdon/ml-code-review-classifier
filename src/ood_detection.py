"""
Out-of-Distribution detection via MSP e Entropy threshold.

MSP (Maximum Softmax Probability): baixa confiança máxima → OOD.
Entropy: H = -sum(p * log(p)). Alta entropia → incerto → OOD.

Thresholds calibrados no val set: percentil 5 da distribuição in-distribution,
garantindo que 95% dos exemplos in-dist passem como in-distribution.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from model import load_finetuned
from train import ReviewDataset

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
MODELS_DIR = Path(__file__).parent.parent / "models"

_OOD_EXAMPLES = [
    # código-fonte puro sem comentário semântico
    "def __init__(self, x, y): self.x = x; self.y = y; return None",
    "SELECT * FROM users WHERE id = 1 ORDER BY created_at DESC LIMIT 10;",
    "import os; import sys; from pathlib import Path; x = Path('/tmp')",
    # frases aleatórias sem relação com code review
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Pizza com abacaxi é uma combinação controversa mas popular no Brasil.",
    "42 is the answer to life, the universe, and everything else.",
    # texto vazio ou quase vazio
    "",
    "   ",
    "???",
    # português misturado com código
    "esse método aqui faz a conexão com banco: conn = db.connect(host)",
    "precisa refatorar isso urgente: for i in range(len(lista)): print(lista[i])",
    "acho que o bug tá aqui: if x = None: pass",
    # jargão não-técnico
    "Please send the quarterly report to all stakeholders by Friday morning.",
    "Meeting rescheduled to 3pm. Agenda: budget review and roadmap discussion.",
]


def compute_entropy(probs: list[float]) -> float:
    """H = -sum(p * log(p)), com p=0 tratado como 0 * log(0) = 0."""
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log(p)
    return h


def is_ood(
    probs: list[float],
    msp_threshold: float = 0.5,
    entropy_threshold: float = 1.2,
) -> dict:
    """
    Detecta OOD usando MSP e Entropy.
    Retorna o método mais restritivo que disparou.
    Se ambos dispararem, retorna o de maior "anomalia" relativa.
    """
    msp = max(probs)
    entropy = compute_entropy(probs)

    msp_ood = msp < msp_threshold
    entropy_ood = entropy > entropy_threshold

    if msp_ood and entropy_ood:
        # retorna o método com anomalia relativa maior
        msp_deviation = (msp_threshold - msp) / msp_threshold
        ent_deviation = (entropy - entropy_threshold) / entropy_threshold
        if msp_deviation >= ent_deviation:
            method = "msp"
            score = msp
            threshold = msp_threshold
        else:
            method = "entropy"
            score = entropy
            threshold = entropy_threshold
    elif msp_ood:
        method = "msp"
        score = msp
        threshold = msp_threshold
    elif entropy_ood:
        method = "entropy"
        score = entropy
        threshold = entropy_threshold
    else:
        method = "none"
        score = msp
        threshold = msp_threshold

    return {
        "is_ood": msp_ood or entropy_ood,
        "method": method,
        "score": score,
        "threshold": threshold,
        "msp": msp,
        "entropy": entropy,
    }


@torch.no_grad()
def _collect_probs(
    checkpoint_path: str,
    split: str,
    batch_size: int = 32,
) -> tuple[list[list[float]], list[int]]:
    device = _get_device()
    model, tokenizer = load_finetuned(checkpoint_path)
    model.to(device)
    model.eval()

    dataset = ReviewDataset(SPLITS_DIR / f"{split}.jsonl", tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs: list[list[float]] = []
    all_labels: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1).cpu()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    return all_probs, all_labels


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calibrate_thresholds(
    checkpoint_path: str,
    val_split: str = "val",
) -> dict:
    """
    Calcula MSP e entropy no val set (in-distribution).
    Threshold = percentil 5 da distribuição → 95% dos in-dist passam.
    Salva em models/ood_thresholds.json.
    """
    probs_list, _ = _collect_probs(checkpoint_path, val_split)

    msps = [max(p) for p in probs_list]
    entropies = [compute_entropy(p) for p in probs_list]

    msps_sorted = sorted(msps)
    entropies_sorted = sorted(entropies)

    n = len(msps_sorted)
    p5_idx = max(0, int(math.floor(0.05 * n)) - 1)
    p95_idx = min(n - 1, int(math.ceil(0.95 * n)))

    msp_threshold = msps_sorted[p5_idx]
    entropy_threshold = entropies_sorted[p95_idx]

    result = {
        "msp_threshold": msp_threshold,
        "entropy_threshold": entropy_threshold,
        "n_samples": n,
        "msp_mean": sum(msps) / n,
        "entropy_mean": sum(entropies) / n,
    }

    out_path = MODELS_DIR / "ood_thresholds.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"MSP threshold (p5):      {msp_threshold:.4f}")
    print(f"Entropy threshold (p95): {entropy_threshold:.4f}")
    print(f"Salvo em {out_path}")

    return result


@torch.no_grad()
def analyze_ood_examples(checkpoint_path: str) -> None:
    """Testa lista hardcoded de exemplos OOD óbvios e imprime resultados."""
    device = _get_device()
    model, tokenizer = load_finetuned(checkpoint_path)
    model.to(device)
    model.eval()

    # tenta carregar thresholds calibrados; fallback nos defaults
    thresholds_path = MODELS_DIR / "ood_thresholds.json"
    if thresholds_path.exists():
        thresholds = json.loads(thresholds_path.read_text())
        msp_thr = thresholds["msp_threshold"]
        ent_thr = thresholds["entropy_threshold"]
    else:
        msp_thr = 0.5
        ent_thr = 1.2
        print("ood_thresholds.json não encontrado — usando defaults (0.5, 1.2)")

    print(f"\nAnálise OOD | MSP threshold={msp_thr:.4f} | Entropy threshold={ent_thr:.4f}")
    print("=" * 80)

    for text in _OOD_EXAMPLES:
        if not text.strip():
            display = repr(text)
        else:
            display = text[:60] + ("..." if len(text) > 60 else "")

        if not text.strip():
            # texto vazio → máxima incerteza por definição
            probs = [1.0 / 5] * 5
        else:
            enc = tokenizer(
                text,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1).cpu().squeeze(0).tolist()

        result = is_ood(probs, msp_threshold=msp_thr, entropy_threshold=ent_thr)

        status = "OOD" if result["is_ood"] else "IN-DIST"
        print(
            f"[{status:7s}] via={result['method']:7s} | "
            f"MSP={result['msp']:.3f} | H={result['entropy']:.3f} | "
            f"text={display}"
        )

    print("=" * 80)


if __name__ == "__main__":
    checkpoint = str(Path(__file__).parent.parent / "models" / "full_finetuned")
    calibrate_thresholds(checkpoint, val_split="val")
    print()
    analyze_ood_examples(checkpoint)
