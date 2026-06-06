# ML Code Review Classifier

POC de machine learning para classificar findings de code review em 5 categorias, usando CodeBERT fine-tuning, LoRA adapters e um dashboard interativo com 11 abas.

Projeto de aprendizado de Milton Abdon do Nascimento Júnior — Software Architect em especialização em IA. O objetivo é passar pelo ciclo completo de ML: dataset → tokenization → training loop manual → avaliação → LoRA → quantização → explainability → active learning.

---

## Resultados

| Técnica | F1 Macro | Params treináveis | Artefato |
|---|---|---|---|
| TF-IDF baseline | 0.824 | — | < 1 MB |
| CodeBERT Full FT | 0.876 | 125M (100%) | 479 MB |
| CodeBERT + LoRA | **0.920** | 1.18M (0.94%) | **7.9 MB** |
| LoRA + INT8 | 0.920 | — | 7.9 MB, −68% RAM |

**Insight chave:** com dataset sintético de 250 exemplos, TF-IDF char ngrams atinge F1=0.82 — delta de apenas 5pp sobre CodeBERT. O valor do pré-treino se manifesta com dados reais e variados.

---

## Classes de classificação

| Label | Descrição |
|---|---|
| `security` | Vulnerabilidades, auth, injection, exposição de dados |
| `architecture` | Acoplamento, coesão, padrões estruturais, SOLID |
| `observability` | Logging ausente, tracing, métricas, alertas |
| `style` | Naming, formatação, magic numbers, dead code |
| `false_positive` | Finding incorreto ou não aplicável neste contexto |

---

## Setup

```bash
# Requer Python 3.11 (PyTorch ainda não suporta 3.14+)
/opt/homebrew/bin/python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

---

## Ordem de execução

### Semana 1 — Dados e fine-tuning completo

```bash
# 1. Gerar dataset (250 exemplos sintéticos, 5 classes balanceadas)
python data/prepare_dataset.py

# 2. Explorar os dados (notebook)
jupyter notebook notebooks/01_exploration.ipynb

# 3. Treinar CodeBERT (fine-tuning completo, ~5 min no MPS)
python src/train.py --epochs 5 --batch-size 8 --lr 2e-5

# 4. Avaliar no test set
python src/evaluate.py

# 5. Ver experimentos no MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Semana 2 — LoRA, calibração e análise

```bash
# 6. Fine-tuning com LoRA (0.94% dos parâmetros)
python src/lora_train.py --epochs 5 --batch-size 8

# 7. Calibração de temperatura (ECE 0.23 → 0.08)
python src/calibration.py

# 8. Calibrar thresholds OOD
python src/ood_detection.py

# 9. Análise de erros
python src/error_analysis.py

# 10. Baseline TF-IDF (quantifica contribuição do CodeBERT)
python src/baseline.py

# 11. Quantização INT8 pós-treino (−68% memória, F1 idêntico)
python src/quantization.py

# 12. Learning curves (quantos dados são necessários?)
python src/learning_curves.py

# 13. Drift detection (monitoramento em produção)
python src/drift_detection.py

# 14. Subir o dashboard
streamlit run dashboard.py
```

### Opcional — Dados reais e pair model

```bash
# Raspar PR comments do GitHub (requer GITHUB_TOKEN)
GITHUB_TOKEN=ghp_xxx python data/github_scraper.py --repos "microsoft/vscode" --max-prs 50

# Pair model: CodeBERT com [CLS] finding [SEP] diff [SEP]
python src/pair_model.py

# Active learning (seleciona exemplos mais incertos para anotação)
python src/active_learning.py
```

---

## Estrutura do projeto

```
ml-code-review-classifier/
├── data/
│   ├── prepare_dataset.py     ← dataset sintético com fallback
│   └── github_scraper.py      ← raspagem de PR comments reais
├── notebooks/
│   ├── 01_exploration.ipynb   ← distribuição, tokenization, exemplos
│   ├── 02_fine_tuning.ipynb   ← training loop interativo + curvas de loss
│   └── 03_lora.ipynb          ← LoRA vs full FT, comparação de parâmetros
├── src/
│   ├── model.py               ← CodeBERT wrapper, load_finetuned()
│   ├── train.py               ← training loop manual (sem Trainer API)
│   ├── lora_train.py          ← PEFT LoRA fine-tuning
│   ├── evaluate.py            ← F1, confusion matrix, top erros
│   ├── calibration.py         ← temperature scaling, ECE, reliability diagram
│   ├── ood_detection.py       ← MSP + entropy threshold
│   ├── error_analysis.py      ← erros por classe, fronteiras de decisão
│   ├── explainability.py      ← Gradient×Input saliency, Attention Rollout
│   ├── quantization.py        ← INT8 pós-treino, benchmark completo
│   ├── baseline.py            ← TF-IDF + LogisticRegression
│   ├── pair_model.py          ← CodeBERT sequence pair (finding + diff)
│   ├── active_learning.py     ← uncertainty sampling, ciclo AL
│   ├── learning_curves.py     ← F1 vs n_exemplos (CodeBERT + baseline)
│   └── drift_detection.py     ← PSI + KS test para monitoramento
├── api/
│   └── classifier_api.py      ← FastAPI v0.2: batch, OOD, calibração
├── tests/
│   ├── conftest.py            ← fixtures de modelo (scope=session)
│   └── test_poc.py            ← 15 testes de comportamento
├── dashboard.py               ← Streamlit 11 abas
├── requirements.txt
└── .gitignore
```

---

## Dashboard (11 abas)

```bash
streamlit run dashboard.py
# Abre em http://localhost:8501
```

| Aba | O que mostra |
|---|---|
| 📈 Experimentos | Curvas de loss/accuracy dos runs MLflow |
| ⚖️ Comparação | Full FT vs LoRA: F1, confusion matrix, tamanho |
| 🎯 Inferência | Classificação ao vivo com badge de confiança |
| 🧪 Simulador | Efeito de hiperparâmetros SEM retreinar |
| 🎯 Calibração | Temperature scaling, ECE, reliability diagram |
| 🚨 OOD | Detecção de inputs fora do domínio |
| 🔬 Erros | Análise de erros por classe, fronteiras de decisão |
| 🧠 Explainability | Saliência por token (Gradient×Input + Rollout) |
| ⚡ Quantização | Benchmark INT8: F1/latência/memória |
| 📊 Baseline | TF-IDF vs CodeBERT, veredicto de justificativa |
| 🔄 Active Learning | Uncertainty sampling, exemplos mais informativos |

---

## API (FastAPI v0.2)

```bash
uvicorn api.classifier_api:app --port 8000

# Classificar um finding
curl -X POST localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"finding": "SQL query built with string concatenation", "model": "lora"}'

# Batch (até 50 findings)
curl -X POST localhost:8000/classify_batch \
  -H "Content-Type: application/json" \
  -d '{"findings": ["...", "..."], "model": "lora"}'

# Health (modelos, OOD, calibração, device)
curl localhost:8000/health

# Metadados do run MLflow
curl localhost:8000/model_info
```

---

## Testes

```bash
pytest tests/ -v
```

15 testes de comportamento cobrindo: predições esperadas, calibração, OOD, baseline e API. Os testes de modelo fazem skip automático se os checkpoints não existirem.

---

## O que esta POC ensina

**Semana 1 — Fine-tuning:**
- Como tokenization transforma texto em tensores
- Por que a loss desce — o que muda nos pesos a cada backward pass
- O que são overfitting e underfitting na prática
- Como MLflow rastreia experimentos

**Semana 2 — LoRA e análise:**
- Como LoRA funciona: rank decomposition, quais camadas adaptar
- Por que partir do checkpoint fine-tunado (não do base model) importa para LoRA
- O que ECE mede e por que confiança ≠ accuracy
- Como Gradient×Input revela o que o modelo "leu" no texto

**Insights não óbvios:**
- TF-IDF char ngrams atinge F1=0.82 com zero GPU — o CodeBERT contribui 5pp com dados sintéticos
- LoRA com 0.94% dos parâmetros supera o full fine-tuning (0.920 vs 0.876)
- INT8 quantização preserva F1 exato com −68% de memória
- `architecture` e `style` têm confiança média ~55% — fronteira semântica real, não ruído
- OOD detection por MSP/entropy falha com dataset sintético homogêneo — precisa de dados reais

---

## Próximos passos naturais

1. **Dados reais**: rodar `github_scraper.py` com `GITHUB_TOKEN` para substituir o dataset sintético
2. **Pair model**: usar `pair_model.py` com diffs reais para classificação contextualizada
3. **Mahalanobis OOD**: mais robusto que MSP/entropy com dados sintéticos
4. **Deploy**: FastAPI + Docker → qualquer cloud com GPU ou CPU quantizado
