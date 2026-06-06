"""
CodeBERT pair model: input = [CLS] finding [SEP] diff [SEP]

CodeBERT foi pré-treinado em pares (NL, Code) — essa estrutura é o uso canônico
do modelo e deve produzir embeddings mais ricos do que texto único para code review.
"""

import json
import argparse
from pathlib import Path

import torch
import mlflow
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from model import load_tokenizer, load_model, LABEL2ID, LABELS, MODEL_NAME

MODELS_DIR = Path(__file__).parent.parent / "models" / "pair_model"

SYNTHETIC_PAIRS = [
    # security — 10 exemplos
    {
        "text": "SQL injection risk in user input concatenation",
        "diff": (
            "def get_user(user_id):\n"
            "    query = 'SELECT * FROM users WHERE id=' + user_id\n"
            "    cursor.execute(query)\n"
            "    return cursor.fetchone()"
        ),
        "label": "security",
    },
    {
        "text": "Hardcoded credentials in source code",
        "diff": (
            "DB_PASSWORD = 'super_secret_123'\n"
            "DB_USER = 'admin'\n"
            "conn = psycopg2.connect(user=DB_USER, password=DB_PASSWORD)"
        ),
        "label": "security",
    },
    {
        "text": "Command injection via unsanitized shell argument",
        "diff": (
            "import subprocess\n"
            "def run_report(filename):\n"
            "    subprocess.run(f'cat {filename}', shell=True)\n"
        ),
        "label": "security",
    },
    {
        "text": "Sensitive data logged in plaintext",
        "diff": (
            "def authenticate(username, password):\n"
            "    logger.info(f'Auth attempt: user={username} pass={password}')\n"
            "    return db.check(username, password)"
        ),
        "label": "security",
    },
    {
        "text": "Insecure deserialization of untrusted data",
        "diff": (
            "import pickle\n"
            "def load_session(data: bytes):\n"
            "    return pickle.loads(data)\n"
        ),
        "label": "security",
    },
    {
        "text": "Missing CSRF token validation on state-changing endpoint",
        "diff": (
            "@app.route('/transfer', methods=['POST'])\n"
            "def transfer():\n"
            "    amount = request.form['amount']\n"
            "    to = request.form['to']\n"
            "    bank.transfer(current_user, to, amount)\n"
            "    return redirect('/')"
        ),
        "label": "security",
    },
    {
        "text": "JWT signature not verified before trusting claims",
        "diff": (
            "def decode_token(token):\n"
            "    # verify=False disables signature check\n"
            "    payload = jwt.decode(token, options={'verify_signature': False})\n"
            "    return payload['user_id']"
        ),
        "label": "security",
    },
    {
        "text": "Path traversal vulnerability in file download",
        "diff": (
            "def download(filename):\n"
            "    path = f'/var/uploads/{filename}'\n"
            "    with open(path, 'rb') as f:\n"
            "        return f.read()"
        ),
        "label": "security",
    },
    {
        "text": "XSS via unescaped user input rendered in template",
        "diff": (
            "# api/views.py\n"
            "def render_profile(name):\n"
            "    # name vem diretamente do request — não escapado\n"
            "    return f'<h1>Welcome {name}</h1>'\n"
        ),
        "label": "security",
    },
    {
        "text": "Open redirect allows phishing via unvalidated URL",
        "diff": (
            "@app.route('/redirect')\n"
            "def redirect_to():\n"
            "    url = request.args.get('next')\n"
            "    return redirect(url)"
        ),
        "label": "security",
    },

    # architecture — 10 exemplos
    {
        "text": "Business logic implemented directly in controller layer",
        "diff": (
            "@GetMapping('/order')\n"
            "public ResponseEntity createOrder(@RequestBody OrderRequest req) {\n"
            "    Order o = new Order(req.getItems());\n"
            "    o.setTotal(req.getItems().stream().mapToDouble(Item::getPrice).sum());\n"
            "    db.save(o);\n"
            "    emailService.sendConfirmation(req.getEmail(), o);\n"
            "    return ResponseEntity.ok(o);\n"
            "}"
        ),
        "label": "architecture",
    },
    {
        "text": "Repository layer directly coupled to HTTP request object",
        "diff": (
            "class UserRepository:\n"
            "    def find_current(self, request):\n"
            "        token = request.headers.get('Authorization')\n"
            "        user_id = jwt.decode(token)['sub']\n"
            "        return self.db.query(User).filter_by(id=user_id).first()"
        ),
        "label": "architecture",
    },
    {
        "text": "Circular dependency between domain modules",
        "diff": (
            "# payment/service.py\n"
            "from order.service import OrderService\n\n"
            "# order/service.py\n"
            "from payment.service import PaymentService\n"
        ),
        "label": "architecture",
    },
    {
        "text": "God class with unrelated responsibilities",
        "diff": (
            "class UserManager:\n"
            "    def register(self): ...\n"
            "    def send_email(self): ...\n"
            "    def process_payment(self): ...\n"
            "    def generate_report(self): ...\n"
            "    def cache_data(self): ...\n"
        ),
        "label": "architecture",
    },
    {
        "text": "Domain entity depends on infrastructure-layer class",
        "diff": (
            "class Order:\n"
            "    def __init__(self, items):\n"
            "        self.items = items\n"
            "        self.repo = SqlAlchemyOrderRepository()\n"
            "    def save(self):\n"
            "        self.repo.persist(self)\n"
        ),
        "label": "architecture",
    },
    {
        "text": "Feature envy: method uses exclusively data from another class",
        "diff": (
            "class Invoice:\n"
            "    def calculate_discount(self, customer):\n"
            "        if customer.loyalty_years > 5:\n"
            "            return customer.base_price * 0.15\n"
            "        return customer.base_price * 0.05\n"
        ),
        "label": "architecture",
    },
    {
        "text": "Anemic domain model with all logic in service layer",
        "diff": (
            "class Order:\n"
            "    pass  # apenas campos, zero comportamento\n\n"
            "class OrderService:\n"
            "    def validate(self, order): ...\n"
            "    def calculate_total(self, order): ...\n"
            "    def apply_discount(self, order): ...\n"
        ),
        "label": "architecture",
    },
    {
        "text": "Synchronous HTTP call inside domain service creates tight coupling",
        "diff": (
            "class ShippingService:\n"
            "    def calculate_fee(self, address):\n"
            "        resp = requests.post('https://logistics.partner/api/fee',\n"
            "                            json={'address': address})\n"
            "        return resp.json()['fee']\n"
        ),
        "label": "architecture",
    },
    {
        "text": "Presentation layer imports persistence model directly",
        "diff": (
            "# api/routes/user.py\n"
            "from db.models import UserORM\n\n"
            "@router.get('/users/{id}')\n"
            "def get_user(id: int):\n"
            "    return session.query(UserORM).get(id)\n"
        ),
        "label": "architecture",
    },
    {
        "text": "Distributed monolith: services share database schema",
        "diff": (
            "# inventory-service/repo.py\n"
            "class InventoryRepo:\n"
            "    def get_order(self, order_id):\n"
            "        # lendo tabela do serviço de pedidos diretamente\n"
            "        return self.session.query(Order).get(order_id)\n"
        ),
        "label": "architecture",
    },

    # observability — 10 exemplos
    {
        "text": "Exception swallowed silently without logging or metrics",
        "diff": (
            "def process_payment(order_id):\n"
            "    try:\n"
            "        payment_gateway.charge(order_id)\n"
            "    except Exception:\n"
            "        pass  # silently ignored\n"
        ),
        "label": "observability",
    },
    {
        "text": "No structured logging — using print statements in production code",
        "diff": (
            "def handle_request(req):\n"
            "    print(f'Received request: {req}')\n"
            "    result = process(req)\n"
            "    print(f'Done: {result}')\n"
            "    return result\n"
        ),
        "label": "observability",
    },
    {
        "text": "Missing correlation ID in logs makes request tracing impossible",
        "diff": (
            "def create_order(items):\n"
            "    logger.info('Creating order')\n"
            "    order = Order(items)\n"
            "    db.save(order)\n"
            "    logger.info('Order created')\n"
            "    return order\n"
        ),
        "label": "observability",
    },
    {
        "text": "Database query latency not measured or logged",
        "diff": (
            "def get_products():\n"
            "    results = db.execute('SELECT * FROM products WHERE active=1')\n"
            "    return results.fetchall()\n"
        ),
        "label": "observability",
    },
    {
        "text": "HTTP client calls external API without timeout or metrics",
        "diff": (
            "def fetch_user_profile(user_id):\n"
            "    response = requests.get(f'{BASE_URL}/users/{user_id}')\n"
            "    return response.json()\n"
        ),
        "label": "observability",
    },
    {
        "text": "Background job has no health check or completion metric",
        "diff": (
            "@celery.task\n"
            "def send_notifications(batch):\n"
            "    for user_id in batch:\n"
            "        notifier.send(user_id)\n"
        ),
        "label": "observability",
    },
    {
        "text": "Log message contains no context about which entity failed",
        "diff": (
            "def update_inventory(items):\n"
            "    for item in items:\n"
            "        try:\n"
            "            inventory.update(item)\n"
            "        except Exception as e:\n"
            "            logger.error('Update failed')\n"
        ),
        "label": "observability",
    },
    {
        "text": "Cache miss rate not tracked — no visibility into cache effectiveness",
        "diff": (
            "def get_config(key):\n"
            "    val = cache.get(key)\n"
            "    if val is None:\n"
            "        val = db.get_config(key)\n"
            "        cache.set(key, val)\n"
            "    return val\n"
        ),
        "label": "observability",
    },
    {
        "text": "Error response returned without logging the underlying cause",
        "diff": (
            "@app.post('/checkout')\n"
            "def checkout(req):\n"
            "    try:\n"
            "        order_service.checkout(req)\n"
            "    except CheckoutError:\n"
            "        return {'error': 'checkout failed'}, 400\n"
        ),
        "label": "observability",
    },
    {
        "text": "Retry logic has no metric on retry count or final failure",
        "diff": (
            "def call_with_retry(fn, retries=3):\n"
            "    for i in range(retries):\n"
            "        try:\n"
            "            return fn()\n"
            "        except Exception:\n"
            "            time.sleep(2 ** i)\n"
            "    raise RuntimeError('all retries failed')\n"
        ),
        "label": "observability",
    },

    # style — 10 exemplos
    {
        "text": "Magic number used without named constant",
        "diff": (
            "def calculate_fee(amount):\n"
            "    if amount > 1000:\n"
            "        return amount * 0.05\n"
            "    return amount * 0.08\n"
        ),
        "label": "style",
    },
    {
        "text": "Function name does not describe what it does",
        "diff": (
            "# pricing/calculator.py\n"
            "def do_stuff(x, y):\n"
            "    # aplica taxa de 10% sobre y e soma com x\n"
            "    return x + y * 1.1\n"
        ),
        "label": "style",
    },
    {
        "text": "Deeply nested conditionals reduce readability",
        "diff": (
            "def process(user):\n"
            "    if user:\n"
            "        if user.active:\n"
            "            if user.verified:\n"
            "                if user.age >= 18:\n"
            "                    return grant_access(user)\n"
        ),
        "label": "style",
    },
    {
        "text": "Dead code left in production — commented-out block",
        "diff": (
            "def send_email(to, subject, body):\n"
            "    # old_mailer.send(to, subject, body)\n"
            "    new_mailer.send(to, subject, body)\n"
        ),
        "label": "style",
    },
    {
        "text": "Variable name is single letter with no obvious meaning",
        "diff": (
            "def compute(l):\n"
            "    r = 0\n"
            "    for e in l:\n"
            "        r += e.v\n"
            "    return r\n"
        ),
        "label": "style",
    },
    {
        "text": "Inconsistent return types in same function",
        "diff": (
            "def find_user(user_id):\n"
            "    user = db.get(user_id)\n"
            "    if user:\n"
            "        return user\n"
            "    return None  # inconsistente com excecao em outros paths\n"
        ),
        "label": "style",
    },
    {
        "text": "Long parameter list — should be replaced with value object",
        "diff": (
            "def create_user(name, email, age, address, city,\n"
            "                country, zip_code, phone, role):\n"
            "    return User(name, email, age, address, city,\n"
            "                country, zip_code, phone, role)\n"
        ),
        "label": "style",
    },
    {
        "text": "Boolean parameter makes call site unreadable",
        "diff": (
            "def send_notification(user, urgent):\n"
            "    ...\n\n"
            "send_notification(user, True)  # what does True mean here?\n"
        ),
        "label": "style",
    },
    {
        "text": "Duplicate code across multiple methods",
        "diff": (
            "def validate_order(order):\n"
            "    if order.total < 0:\n"
            "        raise ValueError('negative total')\n\n"
            "def validate_invoice(invoice):\n"
            "    if invoice.total < 0:\n"
            "        raise ValueError('negative total')\n"
        ),
        "label": "style",
    },
    {
        "text": "Exception type is too broad — catches all exceptions",
        "diff": (
            "try:\n"
            "    result = process(data)\n"
            "except Exception:\n"
            "    logger.error('something went wrong')\n"
        ),
        "label": "style",
    },

    # false_positive — 10 exemplos
    {
        "text": "Intentional use of raw query with parameterized inputs",
        "diff": (
            "def get_user(user_id: int):\n"
            "    # parameterized — não há injection risk\n"
            "    cursor.execute('SELECT * FROM users WHERE id=%s', (user_id,))\n"
            "    return cursor.fetchone()\n"
        ),
        "label": "false_positive",
    },
    {
        "text": "Short variable name is idiomatic in this mathematical context",
        "diff": (
            "# math/linalg.py\n"
            "def dot_product(u, v):\n"
            "    # u, v são vetores — nomenclatura matemática padrão\n"
            "    return sum(a * b for a, b in zip(u, v))\n"
        ),
        "label": "false_positive",
    },
    {
        "text": "Logger call without correlation ID is a test helper, not production path",
        "diff": (
            "# tests/helpers.py\n"
            "def log_test_event(msg):\n"
            "    print(f'[TEST] {msg}')\n"
        ),
        "label": "false_positive",
    },
    {
        "text": "Broad exception catch is intentional in top-level error boundary",
        "diff": (
            "@app.exception_handler(Exception)\n"
            "async def global_handler(request, exc):\n"
            "    logger.exception('Unhandled error', exc_info=exc)\n"
            "    return JSONResponse(status_code=500, content={'error': 'internal'})\n"
        ),
        "label": "false_positive",
    },
    {
        "text": "Direct DB access in repository is correct — this IS the repository layer",
        "diff": (
            "class UserRepository:\n"
            "    def find_by_id(self, user_id: int):\n"
            "        return self.session.query(User).filter_by(id=user_id).first()\n"
        ),
        "label": "false_positive",
    },
    {
        "text": "Hardcoded value is a well-known constant (HTTP port)",
        "diff": (
            "# config/defaults.py\n"
            "# Portas padrão IANA — não são magic numbers\n"
            "DEFAULT_HTTP_PORT = 80\n"
            "DEFAULT_HTTPS_PORT = 443\n"
            "DEFAULT_SSH_PORT = 22\n"
        ),
        "label": "false_positive",
    },
    {
        "text": "Email sent in controller is a thin façade call — service handles business logic",
        "diff": (
            "@router.post('/register')\n"
            "def register(req: RegisterRequest):\n"
            "    user = user_service.register(req)\n"
            "    notification_service.welcome(user)\n"
            "    return user\n"
        ),
        "label": "false_positive",
    },
    {
        "text": "No structured logging in script — intentional, it is a one-shot CLI tool",
        "diff": (
            "#!/usr/bin/env python\n"
            "# scripts/migrate.py — runs once during deploy\n"
            "if __name__ == '__main__':\n"
            "    print('Running migration...')\n"
            "    run_migration()\n"
            "    print('Done.')\n"
        ),
        "label": "false_positive",
    },
    {
        "text": "Method touches another class data because it is a legitimate query object",
        "diff": (
            "class OrderSummaryQuery:\n"
            "    def execute(self, customer):\n"
            "        orders = customer.orders\n"
            "        return [o for o in orders if o.status == 'completed']\n"
        ),
        "label": "false_positive",
    },
    {
        "text": "Retry without metric is acceptable in a test fixture helper",
        "diff": (
            "# tests/fixtures.py\n"
            "def wait_for_service(url, retries=5):\n"
            "    for _ in range(retries):\n"
            "        try:\n"
            "            requests.get(url, timeout=1)\n"
            "            return\n"
            "        except Exception:\n"
            "            time.sleep(1)\n"
        ),
        "label": "false_positive",
    },
]


def generate_synthetic_pairs() -> list:
    return SYNTHETIC_PAIRS


class PairReviewDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = 512):
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
        text = ex["text"]
        diff = ex.get("diff", "")

        # Pair encoding: [CLS] text [SEP] diff [SEP]
        # Exatamente como CodeBERT foi pré-treinado em (NL, Code) pairs.
        encoding = self.tokenizer(
            text,
            diff,
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


def load_pair_model():
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME)
    assert config.max_position_embeddings == 514, (
        f"Esperado max_position_embeddings=514 para CodeBERT, got {config.max_position_embeddings}. "
        "Sequências acima de 512 tokens serão truncadas."
    )
    return load_model(pretrained=True)


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_epoch(model, loader, optimizer, scheduler, device):
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
def _eval_epoch(model, loader, device):
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


def train_pair_model(
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    patience: int = 3,
    train_path: Path = None,
    val_path: Path = None,
    synthetic_fallback: bool = True,
):
    """
    Training loop para o pair model.

    Se train_path/val_path não existirem e synthetic_fallback=True,
    usa os 50 exemplos sintéticos para smoke-test do pipeline.
    Dados sintéticos não são suficientes para produção — use splits reais.
    """
    import tempfile

    device = _get_device()
    print(f"Device: {device}")

    tokenizer = load_tokenizer()
    model = load_pair_model()
    model.to(device)

    splits_dir = Path(__file__).parent.parent / "data" / "splits"
    _train_path = train_path or splits_dir / "train_pairs.jsonl"
    _val_path = val_path or splits_dir / "val_pairs.jsonl"

    tmpfiles = []
    if not _train_path.exists() or not _val_path.exists():
        if not synthetic_fallback:
            raise FileNotFoundError(
                f"Dataset não encontrado: {_train_path} / {_val_path}. "
                "Use synthetic_fallback=True para testar com dados sintéticos."
            )
        print("AVISO: datasets reais não encontrados — usando exemplos sintéticos (smoke test).")
        pairs = generate_synthetic_pairs()
        # 40 train / 10 val
        train_data, val_data = pairs[:40], pairs[40:]

        tmp_train = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for row in train_data:
            tmp_train.write(json.dumps(row) + "\n")
        tmp_train.close()
        tmpfiles.append(tmp_train.name)

        tmp_val = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for row in val_data:
            tmp_val.write(json.dumps(row) + "\n")
        tmp_val.close()
        tmpfiles.append(tmp_val.name)

        _train_path = Path(tmp_train.name)
        _val_path = Path(tmp_val.name)

    train_ds = PairReviewDataset(_train_path, tokenizer, max_length)
    val_ds = PairReviewDataset(_val_path, tokenizer, max_length)
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
    mlflow.set_experiment("code-review-pair-model")
    with mlflow.start_run():
        mlflow.log_params({
            "model": MODEL_NAME,
            "input_mode": "pair_nl_code",
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
            train_loss, train_acc = _train_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss, val_acc = _eval_epoch(model, val_loader, device)

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

    for f in tmpfiles:
        import os
        os.unlink(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--val-path", type=Path, default=None)
    args = parser.parse_args()

    train_pair_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        patience=args.patience,
        train_path=args.train_path,
        val_path=args.val_path,
    )
