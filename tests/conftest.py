import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

FULL_DIR  = ROOT / "models" / "full_finetuned"
LORA_DIR  = ROOT / "models" / "lora_adapter"
TEMP_PATH = ROOT / "models" / "temperature.json"
OOD_PATH  = ROOT / "models" / "ood_thresholds.json"
BASELINE_PATH = ROOT / "models" / "baseline_results.json"


@pytest.fixture(scope="session")
def full_model_and_tokenizer():
    pytest.importorskip("torch")
    if not FULL_DIR.exists():
        pytest.skip("Checkpoint full_finetuned não encontrado — rode python src/train.py")
    from model import load_finetuned
    model, tokenizer = load_finetuned(str(FULL_DIR))
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def lora_model_and_tokenizer(full_model_and_tokenizer):
    if not LORA_DIR.exists():
        pytest.skip("LoRA adapter não encontrado — rode python src/lora_train.py")
    from peft import PeftModel
    from model import load_finetuned
    base, tok = load_finetuned(str(FULL_DIR))
    model = PeftModel.from_pretrained(base, str(LORA_DIR))
    model.eval()
    return model, tok


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient
    from classifier_api import app
    with TestClient(app) as client:
        yield client
