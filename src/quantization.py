"""
Post-training quantization and benchmarking for CodeReviewClassifier.

Compares Full FP32, LoRA FP32, Full INT8, and LoRA INT8 across:
  - F1 macro on test set
  - Mean inference latency (ms/example)
  - Model memory footprint (MB)
  - Checkpoint disk size (MB)
"""

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))

from model import load_finetuned, load_tokenizer, LABELS, LABEL2ID
from train import ReviewDataset

ROOT = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
FULL_CHECKPOINT = str(ROOT / "models" / "full_finetuned")
LORA_CHECKPOINT = str(ROOT / "models" / "lora_adapter")
QUANTIZED_DIR = ROOT / "models" / "quantized"
BENCHMARK_OUTPUT = ROOT / "models" / "quantization_benchmark.json"

WARMUP_RUNS = 5
LATENCY_RUNS = 50


def quantize_model(model, method: str = "dynamic") -> torch.nn.Module:
    """
    Aplica quantização pós-treino.
    method="dynamic": torch.quantization.quantize_dynamic nas camadas Linear
    Retorna modelo quantizado (CPU only — quantização dinâmica não suporta CUDA/MPS)
    """
    model = model.cpu()

    # qnnpack é o único engine disponível no macOS ARM (Apple Silicon).
    # fbgemm é o padrão em x86 Linux/Windows; sem setar, engine="none" → RuntimeError.
    available = torch.backends.quantized.supported_engines
    if "fbgemm" in available:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in available:
        torch.backends.quantized.engine = "qnnpack"
    else:
        raise RuntimeError(f"Nenhum engine de quantização disponível: {available}")

    if method == "dynamic":
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    else:
        raise ValueError(f"Método de quantização não suportado: {method}")
    return quantized


def _load_lora_merged(checkpoint_path: str, lora_path: str):
    """Carrega base model + adapter LoRA e faz merge dos pesos."""
    from peft import PeftModel

    base_model, tokenizer = load_finetuned(checkpoint_path)
    base_model.eval()

    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    return merged_model, tokenizer


def _model_memory_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2


def _checkpoint_disk_mb(path: str) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / 1024 ** 2


def _measure_latency_ms(model, tokenizer, examples: list[dict], n_runs: int = LATENCY_RUNS) -> float:
    """
    Mede latência média de inferência em ms/exemplo.
    Usa batch=1, CPU, descarta os primeiros WARMUP_RUNS runs.
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Pré-tokenizar todos os exemplos para evitar custo de tokenização no loop
    encoded = []
    for ex in examples:
        enc = tokenizer(
            ex["text"],
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded.append({
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        })

    total_runs = WARMUP_RUNS + n_runs
    latencies = []

    with torch.no_grad():
        for i in range(total_runs):
            enc = encoded[i % len(encoded)]
            t0 = time.perf_counter()
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            t1 = time.perf_counter()
            if i >= WARMUP_RUNS:
                latencies.append((t1 - t0) * 1000)

    return sum(latencies) / len(latencies)


@torch.no_grad()
def _compute_f1(model, tokenizer, test_path: Path) -> float:
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    dataset = ReviewDataset(test_path, tokenizer)
    loader = DataLoader(dataset, batch_size=32)

    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    return f1_score(all_labels, all_preds, average="macro")


def _load_test_examples() -> list[dict]:
    examples = []
    with open(SPLITS_DIR / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def benchmark(checkpoint_path: str = FULL_CHECKPOINT, lora_path: str | None = LORA_CHECKPOINT) -> dict:
    """
    Compara configurações FP32 e INT8 para full fine-tuning e (opcionalmente) LoRA.

    Retorna dict com resultados por configuração.
    """
    test_path = SPLITS_DIR / "test.jsonl"
    test_examples = _load_test_examples()

    results = {}

    # --- 1. Full FP32 ---
    print("Carregando Full FP32...")
    full_model, full_tokenizer = load_finetuned(checkpoint_path)
    full_model.cpu().eval()

    print("  Medindo F1...")
    full_f1 = _compute_f1(full_model, full_tokenizer, test_path)
    print("  Medindo latência...")
    full_latency = _measure_latency_ms(full_model, full_tokenizer, test_examples)
    full_mem = _model_memory_mb(full_model)
    full_disk = _checkpoint_disk_mb(checkpoint_path)

    results["Full FP32"] = {
        "f1_macro": round(full_f1, 4),
        "latency_ms": round(full_latency, 2),
        "memory_mb": round(full_mem, 2),
        "disk_mb": round(full_disk, 2),
    }
    print(f"  F1={full_f1:.4f}  lat={full_latency:.1f}ms  mem={full_mem:.1f}MB  disk={full_disk:.1f}MB")

    # --- 2. LoRA FP32 ---
    if lora_path and Path(lora_path).exists():
        print("Carregando LoRA FP32 (merge)...")
        lora_model, lora_tokenizer = _load_lora_merged(checkpoint_path, lora_path)
        lora_model.cpu().eval()

        print("  Medindo F1...")
        lora_f1 = _compute_f1(lora_model, lora_tokenizer, test_path)
        print("  Medindo latência...")
        lora_latency = _measure_latency_ms(lora_model, lora_tokenizer, test_examples)
        lora_mem = _model_memory_mb(lora_model)
        lora_disk = _checkpoint_disk_mb(lora_path)

        results["LoRA FP32"] = {
            "f1_macro": round(lora_f1, 4),
            "latency_ms": round(lora_latency, 2),
            "memory_mb": round(lora_mem, 2),
            "disk_mb": round(lora_disk, 2),
        }
        print(f"  F1={lora_f1:.4f}  lat={lora_latency:.1f}ms  mem={lora_mem:.1f}MB  disk={lora_disk:.1f}MB")
    else:
        lora_model = None
        lora_tokenizer = None

    # --- 3. Full INT8 ---
    print("Quantizando Full FP32 → INT8...")
    full_model_q = load_finetuned(checkpoint_path)[0]
    full_model_q = quantize_model(full_model_q, method="dynamic")
    full_model_q.eval()

    print("  Medindo F1...")
    full_q_f1 = _compute_f1(full_model_q, full_tokenizer, test_path)
    print("  Medindo latência...")
    full_q_latency = _measure_latency_ms(full_model_q, full_tokenizer, test_examples)
    full_q_mem = _model_memory_mb(full_model_q)

    results["Full INT8"] = {
        "f1_macro": round(full_q_f1, 4),
        "latency_ms": round(full_q_latency, 2),
        "memory_mb": round(full_q_mem, 2),
        "disk_mb": round(full_disk, 2),  # disco do checkpoint original FP32
    }
    print(f"  F1={full_q_f1:.4f}  lat={full_q_latency:.1f}ms  mem={full_q_mem:.1f}MB  disk={full_disk:.1f}MB")

    # --- 4. LoRA INT8 ---
    if lora_path and Path(lora_path).exists():
        print("Quantizando LoRA merged FP32 → INT8...")
        lora_model_q, _ = _load_lora_merged(checkpoint_path, lora_path)
        lora_model_q = quantize_model(lora_model_q, method="dynamic")
        lora_model_q.eval()

        print("  Medindo F1...")
        lora_q_f1 = _compute_f1(lora_model_q, lora_tokenizer, test_path)
        print("  Medindo latência...")
        lora_q_latency = _measure_latency_ms(lora_model_q, lora_tokenizer, test_examples)
        lora_q_mem = _model_memory_mb(lora_model_q)

        results["LoRA INT8"] = {
            "f1_macro": round(lora_q_f1, 4),
            "latency_ms": round(lora_q_latency, 2),
            "memory_mb": round(lora_q_mem, 2),
            "disk_mb": round(lora_disk, 2),  # disco do adapter LoRA
        }
        print(f"  F1={lora_q_f1:.4f}  lat={lora_q_latency:.1f}ms  mem={lora_q_mem:.1f}MB  disk={lora_disk:.1f}MB")

    return results


def save_quantized(model, path: str = str(QUANTIZED_DIR)) -> None:
    """
    Salva estado do modelo quantizado via torch.save(model.state_dict(), path/model.pt).
    Também salva tokenizer e config para reload.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out / "model.pt")

    # Salvar config e tokenizer do checkpoint base para permitir reload
    base_model, tokenizer = load_finetuned(FULL_CHECKPOINT)
    tokenizer.save_pretrained(str(out))

    # Salvar config do modelo base para reload posterior
    base_model.config.save_pretrained(str(out))

    print(f"Modelo quantizado salvo em: {out}")
    disk_size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024 ** 2
    print(f"Tamanho em disco: {disk_size:.1f} MB")


def _print_table(results: dict) -> None:
    header = f"{'Configuração':<20} | {'F1 Macro':>8} | {'Latência (ms)':>13} | {'Mem (MB)':>8} | {'Disco (MB)':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, r in results.items():
        print(
            f"{name:<20} | {r['f1_macro']:>8.3f} | {r['latency_ms']:>13.1f} | "
            f"{r['memory_mb']:>8.1f} | {r['disk_mb']:>10.1f}"
        )
    print(sep)


if __name__ == "__main__":
    print("=== Benchmark de Quantização ===\n")

    results = benchmark(
        checkpoint_path=FULL_CHECKPOINT,
        lora_path=LORA_CHECKPOINT if Path(LORA_CHECKPOINT).exists() else None,
    )

    print("\n=== Resultados ===")
    _print_table(results)

    # Salvar JSON
    BENCHMARK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {BENCHMARK_OUTPUT}")

    # Salvar modelo quantizado (full INT8)
    print("\nSalvando modelo quantizado (Full INT8)...")
    full_model_q, _ = load_finetuned(FULL_CHECKPOINT)
    full_model_q = quantize_model(full_model_q, method="dynamic")
    save_quantized(full_model_q, str(QUANTIZED_DIR))
