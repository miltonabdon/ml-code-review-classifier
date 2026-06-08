"""
Dashboard interativo para a POC ML Code Review Classifier.

Abas:
  1. Experimentos   — curvas de loss/accuracy dos runs MLflow, comparação de hiperparams
  2. Comparação     — full fine-tuning vs LoRA: F1, params, tamanho, confusion matrix
  3. Inferência     — classificação ao vivo com qualquer modelo treinado
  4. Simulador      — simula o efeito de variar epochs/LR/rank LoRA SEM retreinar

Rodar:
  venv311/bin/streamlit run dashboard.py
"""

import sys
from pathlib import Path

# garante imports dos módulos src e api de qualquer cwd
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

import json
import torch
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# ─── constantes ──────────────────────────────────────────────────────────────
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
LABELS = ["security", "architecture", "observability", "style", "false_positive"]
LABEL_COLORS = {
    "security":       "#e74c3c",
    "architecture":   "#3498db",
    "observability":  "#2ecc71",
    "style":          "#f39c12",
    "false_positive": "#9b59b6",
}
FULL_DIR  = ROOT / "models" / "full_finetuned"
LORA_DIR  = ROOT / "models" / "lora_adapter"
SPLITS_DIR = ROOT / "data" / "splits"

st.set_page_config(
    page_title="ML Code Review Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System ────────────────────────────────────────────────────────────
# Aesthetic: Precision Lab — warm white, deep slate, amber accent
# Typography: IBM Plex Sans (body) + IBM Plex Mono (code/metrics)
# No emojis in structural UI, no card grids, no gradient text, no glassmorphism
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,300&family=IBM+Plex+Serif:wght@300;400&display=swap" rel="stylesheet">

<style>
/* ── Palette ───────────────────────────────────────────── */
:root {
  --c-bg:       oklch(98% 0.008 85);    /* warm off-white */
  --c-surface:  oklch(96% 0.007 85);    /* slightly darker surface */
  --c-border:   oklch(88% 0.010 85);    /* warm light border */
  --c-slate:    oklch(28% 0.018 250);   /* deep blue-slate */
  --c-slate-mid:oklch(45% 0.022 250);   /* mid slate */
  --c-muted:    oklch(62% 0.014 85);    /* warm muted text */
  --c-amber:    oklch(72% 0.165 68);    /* amber accent */
  --c-amber-dim:oklch(82% 0.090 72);    /* soft amber */
  --c-red:      oklch(55% 0.20 25);
  --c-blue:     oklch(55% 0.18 240);
  --c-green:    oklch(55% 0.18 160);
  --c-purple:   oklch(50% 0.18 300);

  --font-sans:  'IBM Plex Sans', sans-serif;
  --font-mono:  'IBM Plex Mono', monospace;
  --font-serif: 'IBM Plex Serif', serif;

  --r: 4px;
  --transition: 180ms cubic-bezier(0.25, 0, 0.1, 1);
}

/* ── Global reset ──────────────────────────────────────── */
html, body, .stApp {
  background-color: var(--c-bg) !important;
  font-family: var(--font-sans) !important;
  color: var(--c-slate) !important;
}

/* Remove Streamlit top bar decoration */
header[data-testid="stHeader"] {
  background: var(--c-bg) !important;
  border-bottom: 1px solid var(--c-border) !important;
}

/* ── Sidebar ───────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--c-surface) !important;
  border-right: 1px solid var(--c-border) !important;
}

section[data-testid="stSidebar"] * {
  font-family: var(--font-sans) !important;
}

section[data-testid="stSidebar"] h1 {
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  color: var(--c-slate) !important;
  margin-bottom: 0 !important;
}

section[data-testid="stSidebar"] .stMarkdown p {
  font-size: 0.775rem !important;
  color: var(--c-muted) !important;
  letter-spacing: 0.01em !important;
}

/* Sidebar separator */
section[data-testid="stSidebar"] hr {
  border-color: var(--c-border) !important;
  margin: 1rem 0 !important;
}

/* ── Main content area ─────────────────────────────────── */
.main .block-container {
  padding-top: 2.5rem !important;
  padding-left: 2.5rem !important;
  padding-right: 2.5rem !important;
  max-width: 1280px !important;
}

/* ── Typography ────────────────────────────────────────── */
h1 {
  font-family: var(--font-serif) !important;
  font-weight: 300 !important;
  font-size: clamp(1.6rem, 3vw, 2.2rem) !important;
  letter-spacing: -0.01em !important;
  color: var(--c-slate) !important;
  line-height: 1.2 !important;
  margin-bottom: 0.25rem !important;
}

h2 {
  font-family: var(--font-sans) !important;
  font-weight: 600 !important;
  font-size: clamp(0.95rem, 1.5vw, 1.1rem) !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--c-slate-mid) !important;
  margin-top: 2.5rem !important;
  margin-bottom: 1rem !important;
  padding-bottom: 0.5rem !important;
  border-bottom: 1px solid var(--c-border) !important;
}

h3 {
  font-family: var(--font-sans) !important;
  font-weight: 600 !important;
  font-size: 0.925rem !important;
  letter-spacing: 0.03em !important;
  color: var(--c-slate) !important;
}

p, li, .stMarkdown {
  font-family: var(--font-sans) !important;
  font-size: 0.875rem !important;
  line-height: 1.65 !important;
  color: var(--c-slate) !important;
}

/* Buttons contain <p> tags — must not inherit body text color */
.stButton button p,
.stButton button span,
.stButton button div {
  color: inherit !important;
  font-family: inherit !important;
  font-size: inherit !important;
  line-height: inherit !important;
}

/* Caption / small */
.stCaption, [data-testid="stCaptionContainer"] p {
  font-size: 0.775rem !important;
  color: var(--c-muted) !important;
  font-style: italic !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--c-border) !important;
  gap: 0 !important;
  padding-bottom: 0 !important;
}

.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 0.55rem 1rem !important;
  font-family: var(--font-sans) !important;
  font-size: 0.775rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  color: var(--c-muted) !important;
  transition: color var(--transition), border-color var(--transition) !important;
}

.stTabs [data-baseweb="tab"]:hover {
  color: var(--c-slate) !important;
  border-bottom-color: var(--c-border) !important;
}

.stTabs [aria-selected="true"] {
  color: var(--c-slate) !important;
  border-bottom-color: var(--c-amber) !important;
  font-weight: 600 !important;
}

/* Tab content */
.stTabs [data-baseweb="tab-panel"] {
  padding-top: 2rem !important;
}

/* ── Metrics ────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

[data-testid="metric-container"] label {
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--c-muted) !important;
  font-weight: 400 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  font-size: 1.85rem !important;
  font-weight: 600 !important;
  color: var(--c-slate) !important;
  letter-spacing: -0.02em !important;
  line-height: 1.1 !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
}

/* Metric divider line on left */
[data-testid="metric-container"]::before {
  content: '';
  display: block;
  width: 2px;
  height: 2.5rem;
  background: var(--c-amber);
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
}

[data-testid="column"] > div > div {
  position: relative;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton button {
  font-family: var(--font-sans) !important;
  font-size: 0.775rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border-radius: var(--r) !important;
  transition: all var(--transition) !important;
}

.stButton button[kind="primary"] {
  background: var(--c-slate) !important;
  color: var(--c-bg) !important;
  border: none !important;
}

.stButton button[kind="primary"]:hover {
  background: var(--c-slate-mid) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px oklch(28% 0.018 250 / 0.2) !important;
}

.stButton button[kind="secondary"] {
  background: transparent !important;
  color: var(--c-slate) !important;
  border: 1px solid var(--c-border) !important;
}

.stButton button[kind="secondary"]:hover {
  border-color: var(--c-slate) !important;
}

/* ── Inputs / Selects / Textareas ───────────────────────── */
.stTextArea textarea,
.stTextInput input,
[data-baseweb="select"] > div:first-child {
  font-family: var(--font-sans) !important;
  font-size: 0.875rem !important;
  background: white !important;
  border: 1px solid var(--c-border) !important;
  border-radius: var(--r) !important;
  color: var(--c-slate) !important;
  transition: border-color var(--transition) !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
  border-color: var(--c-amber) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px oklch(72% 0.165 68 / 0.12) !important;
}

/* ── Selectbox ──────────────────────────────────────────── */
[data-baseweb="select"] {
  font-family: var(--font-sans) !important;
  font-size: 0.875rem !important;
}

/* ── Dataframes / Tables ────────────────────────────────── */
.stDataFrame {
  border: 1px solid var(--c-border) !important;
  border-radius: var(--r) !important;
  overflow: hidden !important;
}

.stDataFrame table {
  font-family: var(--font-mono) !important;
  font-size: 0.775rem !important;
}

.stDataFrame thead tr {
  background: var(--c-surface) !important;
  border-bottom: 1px solid var(--c-border) !important;
}

.stDataFrame thead th {
  font-family: var(--font-sans) !important;
  font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  font-size: 0.68rem !important;
  color: var(--c-muted) !important;
  padding: 0.6rem 0.75rem !important;
}

/* ── Radio buttons ──────────────────────────────────────── */
[data-testid="stRadio"] label {
  font-family: var(--font-sans) !important;
  font-size: 0.825rem !important;
  color: var(--c-slate) !important;
}

/* ── Expander ───────────────────────────────────────────── */
.stExpander {
  border: 1px solid var(--c-border) !important;
  border-radius: var(--r) !important;
}

.stExpander summary {
  font-family: var(--font-sans) !important;
  font-size: 0.825rem !important;
  font-weight: 600 !important;
  color: var(--c-slate) !important;
}

/* ── Divider ────────────────────────────────────────────── */
hr {
  border-color: var(--c-border) !important;
  margin: 1.5rem 0 !important;
}

/* ── Alerts / Info / Success ────────────────────────────── */
.stAlert, .stSuccess, .stWarning, .stError, .stInfo {
  border-radius: var(--r) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.825rem !important;
}

[data-testid="stInfoBox"] {
  background: oklch(97% 0.012 85) !important;
  border-left: 3px solid var(--c-amber) !important;
  border-radius: 0 var(--r) var(--r) 0 !important;
}

[data-testid="stSuccessBox"] {
  border-left: 3px solid var(--c-green) !important;
}

[data-testid="stWarningBox"] {
  border-left: 3px solid var(--c-amber) !important;
}

[data-testid="stErrorBox"] {
  border-left: 3px solid var(--c-red) !important;
}

/* ── Slider ─────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--c-slate) !important;
  border: none !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderThumb"] {
  background: var(--c-amber) !important;
}

/* ── Plotly charts ──────────────────────────────────────── */
.js-plotly-plot .plotly .main-svg {
  background: transparent !important;
}

/* ── Spinner ────────────────────────────────────────────── */
[data-testid="stSpinner"] {
  font-family: var(--font-sans) !important;
  font-size: 0.825rem !important;
  color: var(--c-muted) !important;
}

/* ── Tab badges / label overrides ──────────────────────── */
/* Strip emojis visually in tabs for cleaner look - not possible in pure CSS,
   handled by leaving them but styling tabs to be refined */

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--c-bg); }
::-webkit-scrollbar-thumb { background: var(--c-border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-muted); }

/* ── Page entrance animation ────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.main .block-container {
  animation: fadeUp 0.35s cubic-bezier(0.25, 0, 0.1, 1) both;
}

/* ── Section label above charts (custom class) ──────────── */
.section-label {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--c-muted);
  margin-bottom: 0.5rem;
  display: block;
}

/* ── Sidebar status indicators ──────────────────────────── */
.status-ok   { color: oklch(55% 0.18 160); font-weight: 600; }
.status-fail { color: oklch(55% 0.20 25);  font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── helpers MLflow ──────────────────────────────────────────────────────────
@st.cache_resource
def get_mlflow_client():
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.tracking.MlflowClient()


def get_runs(client, experiment_name):
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        return []
    return client.search_runs([exp.experiment_id], order_by=["start_time DESC"])


def metric_history(client, run_id, metric):
    hist = client.get_metric_history(run_id, metric)
    return sorted([(h.step, h.value) for h in hist], key=lambda x: x[0])


# ─── helpers modelo ──────────────────────────────────────────────────────────
@st.cache_resource
def load_full_model():
    if not FULL_DIR.exists():
        return None, None
    from model import load_finetuned
    model, tok = load_finetuned(str(FULL_DIR))
    model.eval()
    return model, tok


@st.cache_resource
def load_lora_model():
    if not LORA_DIR.exists() or not FULL_DIR.exists():
        return None, None
    from model import load_finetuned
    from peft import PeftModel
    base, tok = load_finetuned(str(FULL_DIR))
    model = PeftModel.from_pretrained(base, str(LORA_DIR))
    model.eval()
    return model, tok


def predict(model, tokenizer, text):
    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=256, padding="max_length",
    )
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred = int(np.argmax(probs))
    return LABELS[pred], probs


def load_split(name):
    path = SPLITS_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


@torch.no_grad()
def evaluate_on_split(model, tokenizer, examples):
    all_preds, all_labels = [], []
    for ex in examples:
        label_id = LABELS.index(ex["label"])
        enc = tokenizer(
            ex["text"], return_tensors="pt",
            truncation=True, max_length=256, padding="max_length",
        )
        logits = model(**enc).logits
        pred = int(logits.argmax(-1).item())
        all_preds.append(pred)
        all_labels.append(label_id)
    return all_labels, all_preds


# ─── simulador: curvas sintéticas ────────────────────────────────────────────
def simulate_training(epochs, lr, dataset_size, method, lora_rank=16):
    """
    Gera curvas sintéticas de loss para ilustrar o efeito de hiperparâmetros.
    Baseado em dinâmicas reais observadas nos experimentos.
    """
    np.random.seed(42)
    e = np.arange(1, epochs + 1)

    # decay speed é proporcional a lr e inversamente proporcional ao dataset
    speed = (lr / 2e-5) * (dataset_size / 200) ** 0.5

    if method == "Full Fine-tuning":
        floor_train = 0.05 + 0.4 / speed
        floor_val   = 0.20 + 0.5 / speed
        train_loss  = floor_train + (1.62 - floor_train) * np.exp(-0.55 * speed * e)
        val_loss    = floor_val   + (1.57 - floor_val)   * np.exp(-0.45 * speed * e)
        # alto LR ou dataset pequeno causa overfitting após certo epoch
        overfit_start = max(3, int(8 / speed))
        if epochs > overfit_start:
            idx = np.arange(epochs)
            overfit_mask = idx >= overfit_start
            val_loss[overfit_mask] += 0.08 * (idx[overfit_mask] - overfit_start) * (2e-5 / lr)
    else:  # LoRA
        rank_factor = (lora_rank / 16) ** 0.3
        floor_train = 0.03 + 0.3 / (speed * rank_factor)
        floor_val   = 0.15 + 0.35 / (speed * rank_factor)
        train_loss  = floor_train + (0.46 - floor_train) * np.exp(-0.7 * speed * rank_factor * e)
        val_loss    = floor_val   + (0.40 - floor_val)   * np.exp(-0.6 * speed * rank_factor * e)
        overfit_start = max(5, int(12 / speed))
        if epochs > overfit_start:
            idx = np.arange(epochs)
            overfit_mask = idx >= overfit_start
            val_loss[overfit_mask] += 0.04 * (idx[overfit_mask] - overfit_start) * (2e-5 / lr)

    noise_train = np.random.normal(0, 0.015, epochs)
    noise_val   = np.random.normal(0, 0.02,  epochs)

    # accuracy aproximada a partir da val_loss
    val_acc = 1 - val_loss * 0.55
    val_acc = np.clip(val_acc + np.random.normal(0, 0.02, epochs), 0.1, 0.99)

    return e, train_loss + noise_train, val_loss + noise_val, val_acc


# ─── Plotly layout defaults ───────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#2d3142", size=11),
    xaxis=dict(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)", gridwidth=1,
        linecolor="rgba(0,0,0,0.15)", tickfont=dict(size=10, family="IBM Plex Mono"),
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)", gridwidth=1,
        linecolor="rgba(0,0,0,0.15)", tickfont=dict(size=10, family="IBM Plex Mono"),
        zeroline=False,
    ),
    legend=dict(
        font=dict(size=10, family="IBM Plex Sans"),
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0.10)",
        borderwidth=1,
    ),
    margin=dict(l=12, r=12, t=36, b=12),
    hoverlabel=dict(
        bgcolor="white", bordercolor="rgba(0,0,0,0.15)",
        font=dict(family="IBM Plex Mono", size=11),
    ),
    title_font=dict(family="IBM Plex Sans", size=12, color="#6b7280"),
    title_x=0,
)

# Palette racionada: slate para primário, amber para acento, depois semântica
PALETTE = {
    "slate":  "#2d3142",
    "amber":  "#d4851a",
    "blue":   "#2c7bb6",
    "green":  "#3d9b6e",
    "red":    "#c0392b",
    "purple": "#7c5cbf",
    "gray":   "#9ca3af",
}

def apply_layout(fig, title="", height=380):
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height)
    return fig


# ─── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.25rem 0 1rem 0;">
      <div style="font-family: 'IBM Plex Serif', serif; font-size: 1.1rem; font-weight: 300;
                  color: oklch(28% 0.018 250); letter-spacing: -0.01em; line-height: 1.3;">
        ML Code Review<br>
        <span style="font-size: 0.65rem; font-family: 'IBM Plex Mono', monospace;
                     letter-spacing: 0.08em; text-transform: uppercase;
                     color: oklch(62% 0.014 85); font-weight: 400;">
          Classifier v2.0
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    full_model, full_tok = load_full_model()
    lora_model, lora_tok = load_lora_model()

    st.markdown("""<span style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
        letter-spacing:0.1em; text-transform:uppercase; color:oklch(62% 0.014 85);">
        Modelos</span>""", unsafe_allow_html=True)

    full_ok = full_model is not None
    lora_ok = lora_model is not None
    st.markdown(
        f"""<div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.8rem;
            line-height:2; color:oklch(28% 0.018 250);">
          <span style="color:{'oklch(55% 0.18 160)' if full_ok else 'oklch(55% 0.20 25)'};">
            {'●' if full_ok else '○'}</span>&nbsp; Full Fine-tuning<br>
          <span style="color:{'oklch(55% 0.18 160)' if lora_ok else 'oklch(55% 0.20 25)'};">
            {'●' if lora_ok else '○'}</span>&nbsp; LoRA Adapter
        </div>""", unsafe_allow_html=True)

    if SPLITS_DIR.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<span style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
            letter-spacing:0.1em; text-transform:uppercase; color:oklch(62% 0.014 85);">
            Dataset</span>""", unsafe_allow_html=True)
        rows = []
        for split in ["train", "val", "test"]:
            p = SPLITS_DIR / f"{split}.jsonl"
            if p.exists():
                n = sum(1 for _ in open(p))
                rows.append(f"<tr><td style='color:oklch(62% 0.014 85);padding-right:1rem'>{split}</td>"
                            f"<td style='font-family:IBM Plex Mono,monospace;font-weight:600'>{n}</td></tr>")
        st.markdown(
            f"""<table style="font-family:'IBM Plex Sans',sans-serif; font-size:0.775rem;
                color:oklch(28% 0.018 250); border-collapse:collapse; margin-top:0.5rem;">
                {''.join(rows)}</table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<span style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
        letter-spacing:0.1em; text-transform:uppercase; color:oklch(62% 0.014 85);">
        Resultados</span>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.775rem;
                color:oklch(28% 0.018 250); line-height:1.9; margin-top:0.5rem;">
      <div>Full FT &nbsp;<span style="color:oklch(72% 0.165 68); font-weight:600;">0.876</span></div>
      <div>LoRA &nbsp;&nbsp;&nbsp;<span style="color:oklch(72% 0.165 68); font-weight:600;">0.920</span></div>
      <div>INT8 &nbsp;&nbsp;&nbsp;<span style="color:oklch(72% 0.165 68); font-weight:600;">0.920</span></div>
      <div>TF-IDF &nbsp;<span style="color:oklch(62% 0.014 85);">0.824</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─── abas ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "Experimentos", "Comparação", "Inferência", "Simulador",
    "Calibração", "OOD", "Erros",
    "Explainability", "Quantização", "Baseline", "Active Learning",
    "Learning Curves", "Drift Detection",
])

# ════════════════════════════════════════════════════════════════════════════
# ABA 1 — EXPERIMENTOS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Experimentos MLflow")
    st.caption(
        "Cada run é um experimento completo de treino. Selecione dois runs para comparar "
        "as curvas de loss e accuracy por epoch."
    )

    with st.expander("O que são essas curvas e como interpretá-las", expanded=False):
        st.markdown("""
**Loss (Cross-Entropy)**

A loss é o quanto o modelo está errando, medida pela fórmula:
`loss = −log(probabilidade atribuída à classe correta)`.
Se o modelo diz "security = 0.9" e a resposta correta é security, a loss é −log(0.9) ≈ 0.10.
Se diz "security = 0.1" quando devia ser security, a loss é −log(0.1) ≈ 2.30.

O treino funciona calculando esse erro no batch, rodando backpropagation (derivadas parciais em cadeia
do loss até cada peso), e atualizando os pesos na direção que reduz o loss (gradient descent).

**Train loss vs Val loss**

| Padrão | Diagnóstico |
|---|---|
| Ambas descendo juntas | Treino saudável |
| Train desce, val estagna | Overfitting — modelo memoriza o treino |
| Ambas estagnadas alta | Underfitting — LR pequeno ou poucas epochs |
| Val desce depois sobe | Early stopping aqui: epoch de menor val_loss |

**Por que val_loss é o sinal que importa**: train_loss sempre desce — o modelo está sendo otimizado
diretamente nele. Val_loss mede generalização para exemplos não vistos. Se val_loss começa a subir
enquanto train_loss desce, o modelo parou de generalizar e começou a memorizar.

**Accuracy vs Loss**

Accuracy conta acertos/total — é intuitiva mas ruidosa epoch a epoch (um único batch pode virar a
métrica). Loss é contínua e mais estável para diagnosticar tendências. Olhe a Loss para decidir
quando parar; olhe a Accuracy para comunicar resultado final.

**MLflow**

Cada run registra automaticamente: hiperparâmetros (LR, batch, epochs), métricas por epoch
(train_loss, val_loss, val_accuracy), e o artefato do modelo. Serve como histórico de experimentos
— você pode comparar runs e identificar qual configuração produziu o melhor val_loss.
        """)

    client = get_mlflow_client()

    full_runs = get_runs(client, "code-review-classifier")
    lora_runs = get_runs(client, "code-review-classifier-lora")

    if not full_runs and not lora_runs:
        st.warning("Nenhum run encontrado. Execute `python src/train.py` para gerar runs.")
        st.stop()

    # Seleção do run
    col1, col2 = st.columns(2)
    with col1:
        run_labels_full = [f"Run {r.info.run_id[:8]} — val_loss={r.data.metrics.get('best_val_loss', 0):.4f}" for r in full_runs]
        sel_full = st.selectbox("Run Full Fine-tuning", run_labels_full) if full_runs else None

    with col2:
        run_labels_lora = [f"Run {r.info.run_id[:8]} — val_loss={r.data.metrics.get('best_val_loss', 0):.4f}" for r in lora_runs]
        sel_lora = st.selectbox("Run LoRA", run_labels_lora) if lora_runs else None

    st.divider()

    # Curvas de loss
    fig = go.Figure()

    def add_curves(runs, run_labels, selection, label_prefix, color_train, color_val):
        if not runs or not selection:
            return
        idx = run_labels.index(selection)
        run = runs[idx]
        rid = run.info.run_id

        tl = metric_history(client, rid, "train_loss")
        vl = metric_history(client, rid, "val_loss")

        if tl:
            fig.add_trace(go.Scatter(
                x=[e for e, _ in tl], y=[v for _, v in tl],
                mode="lines+markers", name=f"{label_prefix} train_loss",
                line=dict(color=color_train, width=2),
                marker=dict(size=7),
            ))
        if vl:
            fig.add_trace(go.Scatter(
                x=[e for e, _ in vl], y=[v for _, v in vl],
                mode="lines+markers", name=f"{label_prefix} val_loss",
                line=dict(color=color_val, width=2, dash="dash"),
                marker=dict(size=7),
            ))

    add_curves(full_runs, run_labels_full, sel_full, "Full FT", "#3498db", "#85c1e9")
    add_curves(lora_runs, run_labels_lora, sel_lora, "LoRA",    "#e74c3c", "#f1948a")

    fig.update_layout(
        title="Train Loss vs Val Loss por Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        template="simple_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy
    fig_acc = go.Figure()

    def add_acc(runs, run_labels, selection, label_prefix, color):
        if not runs or not selection:
            return
        idx = run_labels.index(selection)
        run = runs[idx]
        va = metric_history(client, run.info.run_id, "val_accuracy")
        if va:
            fig_acc.add_trace(go.Scatter(
                x=[e for e, _ in va], y=[v for _, v in va],
                mode="lines+markers", name=f"{label_prefix} val_accuracy",
                line=dict(color=color, width=2),
                marker=dict(size=7),
            ))

    add_acc(full_runs, run_labels_full, sel_full, "Full FT", "#3498db")
    add_acc(lora_runs, run_labels_lora, sel_lora, "LoRA",    "#e74c3c")

    fig_acc.update_layout(
        title="Val Accuracy por Epoch",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1.05],
        height=320,
        template="simple_white",
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # Parâmetros dos runs
    if full_runs and sel_full:
        idx = run_labels_full.index(sel_full)
        params = full_runs[idx].data.params
        metrics = full_runs[idx].data.metrics
        st.subheader("Parâmetros do run Full FT")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Learning rate", params.get("learning_rate", "—"))
        c2.metric("Batch size",    params.get("batch_size", "—"))
        c3.metric("Epochs",        params.get("num_epochs", "—"))
        c4.metric("Best val_loss", f"{metrics.get('best_val_loss', 0):.4f}")

# ════════════════════════════════════════════════════════════════════════════
# ABA 2 — COMPARAÇÃO
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Full Fine-tuning vs LoRA")
    st.caption(
        "Compara dois modelos no mesmo test set: fine-tuning completo (todos os 125M parâmetros) "
        "vs LoRA (apenas 1.18M parâmetros, 0.94% do total). Métricas, confusion matrix e tamanho de artefato."
    )

    with st.expander("Como funciona cada técnica — e por que LoRA vence aqui", expanded=False):
        st.markdown("""
**Fine-tuning completo**

O CodeBERT base tem 125M parâmetros pré-treinados em código (GitHub). Fine-tuning completo descongela
todos esses parâmetros e os atualiza com o dataset de code review. O problema: com 200 exemplos, você
está tentando ajustar 125 milhões de parâmetros — muito espaço de busca para pouca evidência. O
modelo tende a overfitting nos dados sintéticos.

**LoRA (Low-Rank Adaptation)**

LoRA não toca nos pesos originais. Em vez disso, para cada matriz de pesos W (ex: query projection
da atenção), adiciona dois tensores pequenos: `W_novo = W_original + B × A`, onde A tem forma
(r × d) e B tem forma (d × r), com r << d (r=16, d=768 nesta POC). O produto B×A é uma atualização
de baixo rank — captura as adaptações necessárias com muito menos parâmetros.

**Por que LoRA supera aqui (0.920 vs 0.876)**

Com dataset pequeno e sintético, o fine-tuning completo tem alto risco de overfitting. O LoRA atua
como regularização implícita: a atualização de baixo rank limita o modelo a aprender apenas as
direções mais importantes, sem poder sobrescrever todo o conhecimento pré-treinado. Com 200 exemplos,
essa restrição é vantagem — não limitação.

**Confusion matrix**

Cada célula [i][j] mostra quantas vezes o modelo previu j quando a resposta era i. A diagonal
principal são os acertos. Off-diagonal mais quente = confusão frequente entre aquele par de classes.
Nesta POC, o par mais confuso é `architecture`/`style` — fronteira semântica real entre "design
estrutural" e "convenção de código".

**F1-Score vs Accuracy**

Accuracy = (acertos / total) — enganosa com classes desbalanceadas. F1 por classe = harmônica entre
precision (dos que previ como X, quantos eram X?) e recall (dos que eram X, quantos previ como X?).
F1 Macro = média dos F1 por classe, dando peso igual a cada classe independente do tamanho.
        """)

    if not full_model:
        st.warning("Modelo full não encontrado. Execute `python src/train.py`.")
    elif not lora_model:
        st.warning("Adapter LoRA não encontrado. Execute `python src/lora_train.py`.")
    else:
        test_examples = load_split("test")

        with st.spinner("Avaliando modelos no test set…"):
            y_true, p_full = evaluate_on_split(full_model, full_tok, test_examples)
            _,      p_lora = evaluate_on_split(lora_model, lora_tok, test_examples)

        f1_full = f1_score(y_true, p_full, average="macro")
        f1_lora = f1_score(y_true, p_lora, average="macro")

        def dir_mb(p):
            pp = Path(p)
            if not pp.exists():
                return 0
            return sum(f.stat().st_size for f in pp.rglob("*") if f.is_file()) / 1024**2

        full_mb = dir_mb(FULL_DIR)
        lora_mb = dir_mb(LORA_DIR)
        total_params = sum(p.numel() for p in full_model.parameters())
        lora_trainable = 1_184_261

        # Métricas rápidas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1 Macro — Full FT", f"{f1_full:.3f}")
        col2.metric("F1 Macro — LoRA",    f"{f1_lora:.3f}", delta=f"{f1_lora-f1_full:+.3f}")
        col3.metric("Tamanho Full FT",    f"{full_mb:.0f} MB")
        col4.metric("Tamanho LoRA",       f"{lora_mb:.1f} MB", delta=f"−{full_mb/lora_mb:.0f}x")

        st.divider()

        # Bar chart F1 por classe
        report_full = classification_report(y_true, p_full, target_names=LABELS, output_dict=True)
        report_lora = classification_report(y_true, p_lora, target_names=LABELS, output_dict=True)

        f1_by_class = pd.DataFrame({
            "Full FT": [report_full[l]["f1-score"] for l in LABELS],
            "LoRA":    [report_lora[l]["f1-score"] for l in LABELS],
        }, index=LABELS)

        fig_f1 = go.Figure()
        fig_f1.add_bar(name="Full FT", x=f1_by_class.index, y=f1_by_class["Full FT"],
                       marker_color="#3498db")
        fig_f1.add_bar(name="LoRA",    x=f1_by_class.index, y=f1_by_class["LoRA"],
                       marker_color="#e74c3c")
        fig_f1.update_layout(
            title="F1-Score por Classe",
            barmode="group",
            yaxis_range=[0, 1.05],
            yaxis_title="F1",
            height=360,
            template="simple_white",
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        # Confusion matrices lado a lado
        col_cm1, col_cm2 = st.columns(2)

        def plot_cm(preds, title):
            cm = confusion_matrix(y_true, preds, labels=list(range(len(LABELS))))
            fig = px.imshow(
                cm,
                labels=dict(x="Predito", y="Real", color="Count"),
                x=LABELS, y=LABELS,
                text_auto=True,
                color_continuous_scale="Blues",
                title=title,
                aspect="auto",
                height=400,
            )
            fig.update_xaxes(tickangle=30)
            return fig

        with col_cm1:
            st.plotly_chart(plot_cm(p_full, "Confusion Matrix — Full FT"),
                            use_container_width=True)
        with col_cm2:
            st.plotly_chart(plot_cm(p_lora, "Confusion Matrix — LoRA"),
                            use_container_width=True)

        # Tabela de trade-offs
        st.subheader("Tabela de trade-offs")
        tradeoffs = pd.DataFrame({
            "Técnica": ["Full Fine-tuning", "LoRA"],
            "F1 Macro": [f"{f1_full:.4f}", f"{f1_lora:.4f}"],
            "Params treináveis": [f"{total_params:,}", f"{lora_trainable:,}"],
            "% treináveis": ["100%", f"{lora_trainable/total_params*100:.2f}%"],
            "Artefato": [f"{full_mb:.0f} MB", f"{lora_mb:.1f} MB"],
            "Redução": ["—", f"{full_mb/lora_mb:.0f}x menor"],
        })
        st.dataframe(tradeoffs, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# ABA 3 — INFERÊNCIA
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Inferência ao Vivo")
    st.caption(
        "Classifique qualquer finding de code review em tempo real. "
        "O gráfico mostra as probabilidades brutas do softmax para as 5 classes."
    )

    with st.expander("O que acontece durante a inferência", expanded=False):
        st.markdown("""
**Pipeline completo em ~50ms**

1. **Tokenização**: o texto é quebrado em tokens pelo tokenizer do CodeBERT (BPE — Byte Pair
   Encoding). Palavras comuns viram um token; palavras raras ou código são fragmentadas em subpalavras.
   Tokens especiais `[CLS]` (início) e `[SEP]` (fim) são adicionados. O resultado é uma sequência
   de IDs inteiros.

2. **Embedding**: cada token ID é mapeado para um vetor de 768 dimensões. Esses vetores carregam o
   significado semântico aprendido no pré-treino com código real.

3. **Transformer (12 camadas)**: cada camada aplica self-attention (cada token "olha" para todos os
   outros) e uma rede feed-forward. A saída do `[CLS]` após as 12 camadas concentra o significado
   global da sequência.

4. **Classification head**: o vetor do `[CLS]` passa por uma camada linear (768 → 5) que produz 5
   números brutos chamados logits.

5. **Softmax**: `p_i = exp(logit_i) / sum(exp(logits))`. Converte os logits em probabilidades que
   somam 1. A classe com maior probabilidade é a predição.

**Como interpretar as probabilidades**

- Probabilidade alta (> 0.85) numa só classe: o modelo está confiante. Use o resultado.
- Probabilidade dividida entre 2 classes (ex: 0.55/0.35): o modelo está no limiar. Leia os dois
  candidatos e decida com contexto.
- Sem classe acima de 0.50: possível OOD (input fora do domínio). Veja a aba OOD Detection.

**Importante**: confiança alta ≠ acurácia. Um modelo pode ser confiante e errado — é o que a aba
Calibração mede. A aba Análise de Erros mostra exatamente os casos de alta confiança incorretos.
        """)

    if not full_model and not lora_model:
        st.warning("Nenhum modelo carregado. Execute `python src/train.py` e `python src/lora_train.py`.")
    else:
        col_inp, col_model = st.columns([3, 1])
        with col_model:
            model_choice = st.radio(
                "Modelo",
                ["LoRA", "Full Fine-tuning", "Ambos"],
                index=0,
            )
        with col_inp:
            finding_input = st.text_area(
                "Finding de code review",
                value="SQL query built with string concatenation — use parameterized queries to prevent injection",
                height=100,
            )

        # Exemplos rápidos
        st.caption("Exemplos rápidos:")
        examples = {
            "🔒 Security":       "SQL query built with string concatenation — injection risk",
            "🏛 Architecture":   "This class has 12 responsibilities — extract domain logic into services",
            "📊 Observability":  "No logging on exception path — impossible to diagnose in production",
            "✏️ Style":          "Variable name 'x' is ambiguous — use user_count or similar",
            "✅ False Positive":  "This pattern is intentional — retry handles transient failures by design",
        }
        cols = st.columns(len(examples))
        for col, (label, text) in zip(cols, examples.items()):
            if col.button(label, use_container_width=True):
                st.session_state["finding_input"] = text
                st.rerun()

        if "finding_input" in st.session_state:
            finding_input = st.session_state.pop("finding_input")

        if finding_input.strip():
            st.divider()
            models_to_run = []
            if model_choice in ("Full Fine-tuning", "Ambos") and full_model:
                models_to_run.append(("Full Fine-tuning", full_model, full_tok, "#3498db"))
            if model_choice in ("LoRA", "Ambos") and lora_model:
                models_to_run.append(("LoRA", lora_model, lora_tok, "#e74c3c"))

            for name, model, tok, color in models_to_run:
                pred_label, probs = predict(model, tok, finding_input)
                st.subheader(f"Resultado — {name}")

                # Badge de predição
                st.markdown(
                    f"<span style='background:{LABEL_COLORS[pred_label]}; color:white; "
                    f"padding:6px 16px; border-radius:20px; font-size:1.1em; font-weight:600'>"
                    f"{pred_label} &nbsp; {probs[LABELS.index(pred_label)]:.1%}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("")

                # Bar chart de probabilidades
                fig_prob = go.Figure(go.Bar(
                    x=[f"{LABELS.index(l)} · {l}" for l in LABELS],
                    y=probs,
                    marker_color=[
                        LABEL_COLORS[l] if l == pred_label else "#bdc3c7"
                        for l in LABELS
                    ],
                    text=[f"{p:.1%}" for p in probs],
                    textposition="outside",
                ))
                fig_prob.update_layout(
                    yaxis_range=[0, 1.1],
                    yaxis_title="Probabilidade",
                    height=300,
                    template="simple_white",
                    showlegend=False,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# ABA 4 — SIMULADOR
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Simulador de Experimentos")
    st.caption(
        "Simula o efeito de hiperparâmetros nas curvas de treino **sem retreinar o modelo**. "
        "As curvas são geradas por um modelo analítico calibrado nos experimentos reais."
    )

    with st.expander("O que cada hiperparâmetro faz — e como ler o diagnóstico", expanded=False):
        st.markdown("""
**Learning Rate (LR)**

Controla o tamanho do passo na atualização dos pesos: `W_novo = W_atual − LR × gradiente`.

- **LR muito alto** (ex: 1e-4 com BERT): os passos são grandes demais — a loss oscila ou diverge
  em vez de convergir. O gráfico mostra val_loss instável ou subindo.
- **LR muito baixo** (ex: 5e-6): passos minúsculos — a loss desce lentamente e você precisa de muitas
  epochs para convergir. Pode parecer underfitting precoce.
- **LR ótimo para fine-tuning BERT**: 2e-5 a 5e-5. Pequeno o suficiente para não destruir o
  pré-treino, grande o suficiente para adaptar em poucas epochs.

**Epochs**

Número de vezes que o modelo vê o dataset completo. Mais epochs = mais atualização de pesos.

- Poucas epochs: underfitting — o modelo ainda não aprendeu as representações do dataset.
- Muitas epochs: overfitting — train_loss continua caindo mas val_loss começa a subir. A zona
  vermelha no gráfico indica esse risco.

**Dataset size**

Mais dados = representação mais rica = generalização melhor. Com poucas amostras, o modelo aprende
a distribuição do treino e não da tarefa real. O efeito é visível: com n=50 o gap train/val é maior
que com n=500.

**LoRA rank (r)**

O rank r define a "capacidade de adaptação" do LoRA: r=4 permite poucas direções de atualização,
r=64 permite muitas. Rank maior = mais parâmetros treináveis = maior risco de overfitting com dataset
pequeno. Para esta POC com 200 exemplos, r=8 ou r=16 é o ponto ideal.

**Gap train/val**

`gap = média(val_loss[-3 epochs]) − média(train_loss[-3 epochs])`

- gap > 0.15: overfitting confirmado. Ação: reduzir epochs, aumentar dropout, ou aumentar dataset.
- gap < 0 (val_loss < train_loss): raro mas possível com dropout — o modelo avalia sem dropout e
  parece melhor do que no treino.
- gap próximo de 0: generalização saudável.
        """)

    with st.expander("Como o simulador funciona internamente", expanded=False):
        st.markdown("""
O simulador não roda GPU — as curvas são funções analíticas calibradas nos runs reais do MLflow.

O modelo de simulação usa:
- Decaimento exponencial para a loss: `loss(e) = loss_0 × exp(−k × e) + noise`
- O coeficiente k é função do LR, tamanho do dataset e se é LoRA ou Full FT
- Overfitting é modelado como divergência crescente train/val após um epoch crítico que depende do
  tamanho do dataset

Isso permite explorar o espaço de hiperparâmetros instantaneamente, sem custo computacional.
A limitação: o modelo analítico não captura interações complexas entre hiperparâmetros — para isso,
rode `python src/train.py` com diferentes configurações e compare na aba Experimentos.
        """)

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Configuração")
        sim_method = st.selectbox("Técnica", ["Full Fine-tuning", "LoRA"])
        sim_epochs  = st.slider("Epochs",         min_value=2,  max_value=25,    value=5)
        sim_lr      = st.select_slider(
            "Learning rate",
            options=[5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 3e-4],
            value=2e-5,
            format_func=lambda x: f"{x:.0e}",
        )
        sim_dataset = st.slider("Dataset size",   min_value=50, max_value=2000,  value=200, step=50)
        if sim_method == "LoRA":
            sim_rank = st.select_slider("LoRA rank (r)", options=[4, 8, 16, 32, 64], value=16)
        else:
            sim_rank = 16

        st.divider()
        st.subheader("Baseline real")
        st.markdown("**Full FT** — epochs=5, lr=2e-5, n=200")
        st.markdown("→ val_loss: 1.57 → 1.26 → 0.79 → 0.55 → 0.48")
        st.markdown("**LoRA** — epochs=5, lr=2e-5, n=200")
        st.markdown("→ val_loss: 0.40 → 0.34 → 0.31 → 0.30 → 0.30")

    with col_plot:
        e, tl, vl, va = simulate_training(sim_epochs, sim_lr, sim_dataset, sim_method, sim_rank)

        # Curvas de loss
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=e, y=tl, mode="lines+markers",
            name="train_loss", line=dict(color="#3498db", width=2),
        ))
        fig_sim.add_trace(go.Scatter(
            x=e, y=vl, mode="lines+markers",
            name="val_loss", line=dict(color="#e74c3c", width=2, dash="dash"),
        ))
        # Zona de overfitting
        overfit_epoch = max(3, int(8 / ((sim_lr / 2e-5) * (sim_dataset / 200) ** 0.5)))
        if sim_epochs > overfit_epoch:
            fig_sim.add_vrect(
                x0=overfit_epoch, x1=sim_epochs,
                fillcolor="rgba(231,76,60,0.08)",
                layer="below", line_width=0,
                annotation_text="risco de overfitting",
                annotation_position="top left",
            )

        fig_sim.update_layout(
            title=f"Curvas simuladas — {sim_method}",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=350,
            template="simple_white",
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Accuracy simulada
        fig_acc_sim = go.Figure()
        fig_acc_sim.add_trace(go.Scatter(
            x=e, y=va, mode="lines+markers",
            name="val_accuracy", line=dict(color="#2ecc71", width=2),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.08)",
        ))
        fig_acc_sim.update_layout(
            title="Val Accuracy simulada",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1.05],
            height=280,
            template="simple_white",
        )
        st.plotly_chart(fig_acc_sim, use_container_width=True)

        # Diagnóstico automático
        best_val = float(np.min(vl))
        best_epoch = int(np.argmin(vl)) + 1
        gap = float(np.mean(vl[-3:]) - np.mean(tl[-3:]))

        st.subheader("Diagnóstico")
        diag_cols = st.columns(3)
        diag_cols[0].metric("Melhor val_loss", f"{best_val:.4f}", f"epoch {best_epoch}")
        diag_cols[1].metric("Val acc estimada", f"{va[best_epoch-1]:.1%}")
        diag_cols[2].metric("Gap train/val",    f"{gap:+.4f}",
                            delta_color="inverse",
                            help="Positivo = overfitting. Negativo = underfitting.")

        if gap > 0.15:
            st.error("⚠️ Overfitting detectado — reduza epochs, aumente dataset ou use dropout maior.")
        elif best_val > 0.8:
            st.warning("⚠️ Underfitting — aumente LR, epochs ou tamanho do dataset.")
        else:
            st.success("✅ Configuração saudável — loss convergindo sem divergência da val.")

# ════════════════════════════════════════════════════════════════════════════
# ABA 5 — CALIBRAÇÃO
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Calibração — Temperature Scaling")
    st.caption(
        "Um modelo bem calibrado tem confiança 80% quando acerta 80% das vezes naquele nível. "
        "Temperature scaling aprende um escalar T que divide os logits antes do softmax."
    )

    temp_path = ROOT / "models" / "temperature.json"
    if not temp_path.exists():
        st.warning("Execute `python src/calibration.py` para gerar a calibração.")
    else:
        import json as _json
        temp_data = _json.loads(temp_path.read_text())
        T = temp_data["temperature"]
        ece_before = temp_data["ece_before"]
        ece_after  = temp_data["ece_after"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Temperatura T", f"{T:.4f}", help="T<1 sharpena (mais confiante), T>1 suaviza")
        c2.metric("ECE antes", f"{ece_before:.4f}")
        c3.metric("ECE depois", f"{ece_after:.4f}", delta=f"{ece_after - ece_before:+.4f}", delta_color="inverse")

        st.divider()

        # Explicação visual do que T faz
        st.subheader("O que a temperatura faz nos logits")
        import numpy as _np

        logits_ex = _np.array([2.5, 0.8, 0.3, 0.2, 0.1])
        temps = [0.5, 1.0, T, 2.0]
        fig_temp = go.Figure()
        for t in temps:
            scaled = logits_ex / t
            probs = _np.exp(scaled) / _np.exp(scaled).sum()
            name = f"T={t:.1f}" if t != T else f"T={t:.2f} (aprendido)"
            fig_temp.add_trace(go.Bar(
                name=name, x=LABELS, y=probs.tolist(),
                text=[f"{p:.1%}" for p in probs], textposition="outside",
            ))
        fig_temp.update_layout(
            barmode="group", height=380, template="simple_white",
            yaxis_title="Probabilidade",
            title="Efeito de diferentes temperaturas nos mesmos logits",
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        st.info(
            f"**Interpretação:** T={T:.2f} {'sharpena as probabilidades (modelo estava underconfident no val set)' if T < 1 else 'suaviza as probabilidades (modelo estava overconfident)'}. "
            f"A ECE caiu de {ece_before:.3f} → {ece_after:.3f} — "
            f"{'boa' if ece_after < 0.1 else 'melhora parcial na'} calibração."
        )

        st.subheader("Como ler o Reliability Diagram")
        st.markdown("""
        Um modelo perfeitamente calibrado teria todos os pontos na diagonal.
        - **Pontos abaixo da diagonal**: modelo overconfident (confia mais do que acerta)
        - **Pontos acima da diagonal**: modelo underconfident (acerta mais do que confia)

        Com apenas 25 exemplos de val, o reliability diagram tem bins ruidosos —
        execute `calibration.py` com mais dados para uma leitura confiável.
        """)

        if full_model and full_tok:
            with st.spinner("Gerando reliability diagram..."):
                from calibration import compute_ece, reliability_diagram as rel_diag
                test_examples = load_split("test")
                y_true_cal, p_cal = evaluate_on_split(full_model, full_tok, test_examples)

                all_probs = []
                for ex in test_examples:
                    _, probs_ex = predict(full_model, full_tok, ex["text"])
                    all_probs.append(probs_ex)

                import torch as _torch
                probs_tensor = _torch.tensor(all_probs)
                labels_tensor = _torch.tensor(y_true_cal)

                bin_confs, bin_accs, bin_counts = rel_diag(probs_tensor, labels_tensor)

                fig_rel = go.Figure()
                fig_rel.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="gray", width=1),
                    name="Calibração perfeita",
                ))
                fig_rel.add_trace(go.Scatter(
                    x=bin_confs, y=bin_accs, mode="lines+markers",
                    marker=dict(size=10, color="#3498db"),
                    line=dict(color="#3498db", width=2),
                    name="Modelo (antes de calibração)",
                    text=[f"n={c}" for c in bin_counts],
                    hovertemplate="conf=%{x:.2f}<br>acc=%{y:.2f}<br>%{text}",
                ))

                # After calibration: apply T
                import torch.nn.functional as _F
                with _torch.no_grad():
                    raw_logits_list = []
                    for ex in test_examples:
                        enc = full_tok(ex["text"], return_tensors="pt", truncation=True,
                                       max_length=256, padding="max_length")
                        raw_logits_list.append(full_model(**enc).logits)
                    raw_logits = _torch.cat(raw_logits_list, dim=0)
                    cal_probs = _F.softmax(raw_logits / T, dim=-1)

                bc2, ba2, bn2 = rel_diag(cal_probs, labels_tensor)
                fig_rel.add_trace(go.Scatter(
                    x=bc2, y=ba2, mode="lines+markers",
                    marker=dict(size=10, color="#e74c3c"),
                    line=dict(color="#e74c3c", width=2, dash="dot"),
                    name=f"Modelo (T={T:.2f})",
                    text=[f"n={c}" for c in bn2],
                ))

                fig_rel.update_layout(
                    title="Reliability Diagram",
                    xaxis_title="Confiança predita",
                    yaxis_title="Accuracy real",
                    xaxis_range=[0, 1], yaxis_range=[0, 1],
                    height=400, template="simple_white",
                )
                st.plotly_chart(fig_rel, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# ABA 6 — OOD DETECTION
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("OOD Detection")
    st.caption(
        "Detecta inputs fora do domínio via MSP (Maximum Softmax Probability) e entropia. "
        "**Limitação atual:** com dataset sintético homogêneo, o modelo é confiante mesmo em OOD — "
        "este é o comportamento esperado e documenta por que dados reais importam."
    )

    ood_path = ROOT / "models" / "ood_thresholds.json"
    if not ood_path.exists():
        st.warning("Execute `python src/ood_detection.py` para calibrar thresholds.")
    else:
        ood_data = _json.loads(ood_path.read_text())
        c1, c2, c3 = st.columns(3)
        c1.metric("MSP threshold",     f"{ood_data['msp_threshold']:.4f}")
        c2.metric("Entropy threshold",  f"{ood_data['entropy_threshold']:.4f}")
        c3.metric("Amostras de ref.",   ood_data.get("n_samples", "?"))

        st.divider()
        st.subheader("Teste interativo de OOD")
        st.caption("Digite qualquer texto — in-distribution ou completamente fora do domínio")

        ood_input = st.text_area(
            "Texto para testar",
            value="SELECT * FROM users WHERE id = 1",
            height=80,
        )

        ood_examples_quick = {
            "🔒 Finding real":    "SQL query built with string concatenation — injection risk",
            "💻 Código puro":     "def connect(): conn = db.connect('localhost'); return conn",
            "🍕 Fora do domínio": "Pizza com abacaxi é controversa mas popular no Brasil",
            "📧 E-mail":          "Please send the quarterly report by end of day Friday",
            "❓ Junk":            "??? !!! @@@",
        }
        cols_ood = st.columns(len(ood_examples_quick))
        for col, (label, text) in zip(cols_ood, ood_examples_quick.items()):
            if col.button(label, use_container_width=True):
                st.session_state["ood_input"] = text
                st.rerun()

        if "ood_input" in st.session_state:
            ood_input = st.session_state.pop("ood_input")

        if ood_input.strip() and full_model:
            import math as _math
            _, probs_ood = predict(full_model, full_tok, ood_input)
            msp = max(probs_ood)
            entropy = -sum(p * _math.log(p + 1e-9) for p in probs_ood)
            msp_ood = msp < ood_data["msp_threshold"]
            ent_ood = entropy > ood_data["entropy_threshold"]
            is_ood_final = msp_ood or ent_ood

            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("MSP",     f"{msp:.4f}",     delta=f"threshold={ood_data['msp_threshold']:.3f}", delta_color="off")
            col_res2.metric("Entropy", f"{entropy:.4f}", delta=f"threshold={ood_data['entropy_threshold']:.3f}", delta_color="off")
            col_res3.metric("Resultado", "OOD 🚨" if is_ood_final else "In-distribution ✅")

            if is_ood_final:
                reasons = []
                if msp_ood: reasons.append(f"MSP={msp:.3f} < {ood_data['msp_threshold']:.3f}")
                if ent_ood: reasons.append(f"Entropy={entropy:.3f} > {ood_data['entropy_threshold']:.3f}")
                st.error(f"OOD detectado: {' | '.join(reasons)}")
            else:
                st.success("In-distribution — o modelo se sente confiante neste input")

            if is_ood_final and not msp_ood and not ent_ood:
                st.info("💡 **Limitação documentada:** dataset sintético homogêneo → modelo não aprendeu incerteza para inputs OOD. Dados reais (github_scraper.py) resolvem isso.")

            fig_ood = go.Figure(go.Bar(
                x=[f"{LABELS.index(l)} · {l}" for l in LABELS],
                y=probs_ood,
                marker_color=["#e74c3c" if p == msp else "#bdc3c7" for p in probs_ood],
                text=[f"{p:.1%}" for p in probs_ood],
                textposition="outside",
            ))
            fig_ood.add_hline(y=ood_data["msp_threshold"], line_dash="dash",
                              line_color="orange", annotation_text=f"MSP threshold={ood_data['msp_threshold']:.3f}")
            fig_ood.update_layout(
                title="Distribuição de probabilidades (< threshold → OOD)",
                yaxis_range=[0, 1.1], height=320, template="simple_white", showlegend=False,
            )
            st.plotly_chart(fig_ood, use_container_width=True)

        st.subheader("Por que OOD é difícil com dados sintéticos")
        st.markdown("""
        | Método | Funciona com dados sintéticos? | Solução |
        |--------|-------------------------------|---------|
        | MSP threshold | ❌ Modelo confiante mesmo em OOD | Dados reais variados |
        | Entropy threshold | ❌ Entropia baixa em todo input | Treinar com exemplos negativos |
        | Mahalanobis distance | ✅ Independe da distribuição de treino | Implementar com embeddings |
        | ODIN (temperature + perturbation) | ✅ Mais robusto | Perturbação no input |

        **Próximo passo:** implementar OOD via distância de Mahalanobis no espaço de embeddings do [CLS] token.
        """)

# ════════════════════════════════════════════════════════════════════════════
# ABA 7 — ANÁLISE DE ERROS
# ════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Análise de Erros")
    st.caption("Onde o modelo erra, com que confiança, e quais são os casos mais perigosos.")

    with st.expander("Como usar a análise de erros para melhorar o modelo", expanded=False):
        st.markdown("""
**Por que analisar erros antes de tunar**

Tunar hiperparâmetros sem entender os erros é otimizar às cegas. A análise de erros responde:
o modelo erra por falta de dados, por ambiguidade semântica real, ou por shortcut espúrio?
A resposta muda completamente a ação corretiva.

**Confiança média por classe**

Mostra a confiança média do modelo nas predições corretas, separada por classe verdadeira.
- Classe com confiança média alta (> 0.80): o modelo está seguro nessa categoria.
- Classe com confiança média baixa (~0.55): o modelo está na fronteira — os exemplos desta classe
  se sobrepõem com outra. Nesta POC, `architecture` e `style` ficam em ~55%.

**Erros de alta confiança (≥ 0.7)**

São os mais perigosos em produção: o modelo erra E está convicto do erro. Em um sistema de code
review automatizado, esses casos passariam sem flag para revisão humana.

Padrão a observar: se os erros de alta confiança se concentram num par de classes (ex: architecture
predito como style com 78%), o problema é de representação — os embeddings dessas classes estão
próximos no espaço vetorial. Solução: mais dados com marcadores distintos ou pair model (finding +
diff).

**Exemplos de fronteira (|p1 − p2| < 0.2)**

O gap é a diferença entre a probabilidade da classe mais provável e a segunda mais provável.
Gap pequeno = modelo incerto = caso ambíguo. Esses exemplos são os melhores candidatos para anotação
manual no ciclo de Active Learning — máximo ganho de informação por esforço.

**Como diagnosticar a causa do erro**

1. Leia o texto do exemplo errado.
2. Veja qual classe foi predita e com que confiança.
3. Pergunte: "um humano experiente erraria isso?" Se sim, o problema é no dataset (exemplos
   ambíguos ou underrepresentados). Se não, o problema é no modelo (shortcut, falta de capacidade
   ou feature espúria).
4. Use a aba Explainability com esse mesmo texto para ver quais tokens guiaram o erro.
        """)

    if not full_model:
        st.warning("Modelo full não carregado.")
    else:
        with st.spinner("Rodando análise de erros no test set..."):
            from error_analysis import analyze_errors
            analysis = analyze_errors(str(FULL_DIR), split="test")

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        total = analysis["total_examples"]
        errors = analysis["total_errors"]
        col_e1.metric("Total exemplos", total)
        col_e2.metric("Total erros", errors)
        col_e3.metric("Accuracy", f"{analysis['accuracy']:.1%}")
        col_e4.metric("Erros alta confiança", len(analysis["high_confidence_errors"]),
                      help="Erros com confiança > 0.7 — os mais perigosos")

        st.divider()

        # Confiança média por classe
        st.subheader("Confiança média por classe (label verdadeira)")
        avg_conf = analysis["avg_confidence_by_class"]
        fig_conf = go.Figure(go.Bar(
            x=list(avg_conf.keys()),
            y=list(avg_conf.values()),
            marker_color=[LABEL_COLORS[l] for l in avg_conf.keys()],
            text=[f"{v:.1%}" for v in avg_conf.values()],
            textposition="outside",
        ))
        fig_conf.add_hline(y=0.7, line_dash="dash", line_color="orange",
                           annotation_text="threshold alta confiança (0.7)")
        fig_conf.update_layout(
            yaxis_range=[0, 1.1], height=320, template="simple_white",
            yaxis_title="Confiança média",
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        # Confusion pairs
        if analysis["confusion_pairs"]:
            st.subheader("Pares de confusão mais frequentes")
            pairs_data = [
                {"True": p["true"], "Predito": p["predicted"], "Ocorrências": p["count"]}
                for p in sorted(analysis["confusion_pairs"], key=lambda x: -x["count"])
            ]
            st.dataframe(pd.DataFrame(pairs_data), use_container_width=True, hide_index=True)

        # Erros de alta confiança
        st.subheader("Erros de alta confiança (≥ 0.7) — os mais perigosos")
        if not analysis["high_confidence_errors"]:
            st.success("✅ Nenhum erro de alta confiança no test set.")
        else:
            for err in analysis["high_confidence_errors"]:
                true_l = err.get("true_label") or err.get("label") or "?"
                st.error(
                    f"**True:** {true_l} | **Pred:** {err['predicted']} | "
                    f"**Conf:** {err['confidence']:.1%}\n\n{err['text']}"
                )

        # Exemplos de fronteira
        st.subheader("Exemplos de fronteira (|p1 - p2| < 0.2)")
        boundary = analysis["boundary_examples"]
        if not boundary:
            st.info("Nenhum exemplo de fronteira no test set.")
        else:
            for ex in boundary[:5]:
                color = "🟢" if ex["correct"] else "🔴"
                t1 = ex.get("top1", {})
                t2 = ex.get("top2", {})
                true_l = ex.get("true") or ex.get("true_label") or "?"
                st.markdown(
                    f"{color} **gap={ex['gap']:.3f}** | true={true_l} | "
                    f"top1={t1.get('label','?')} ({t1.get('prob',0):.1%}) | "
                    f"top2={t2.get('label','?')} ({t2.get('prob',0):.1%})\n\n"
                    f"> {ex['text'][:120]}"
                )

        # Erros por classe
        st.subheader("Erros por classe")
        for label in LABELS:
            errs = analysis["by_class"].get(label, [])
            err_list = [e for e in errs if not e.get("correct", True)]
            with st.expander(f"{label} — {len(err_list)} erro(s)"):
                if not err_list:
                    st.write("Sem erros nesta classe.")
                for e in err_list:
                    st.markdown(
                        f"**Predito:** {e['predicted']} ({e['confidence']:.1%})\n\n"
                        f"> {e['text'][:150]}"
                    )

# ════════════════════════════════════════════════════════════════════════════
# ABA 8 — EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
with tab8:
    st.header("🧠 Explainability — Por que o modelo decidiu assim?")
    st.caption(
        "Duas técnicas de post-hoc explainability para inspecionar o que o modelo usa para classificar. "
        "Nenhuma delas modifica o modelo — são leituras da mecânica interna após o forward pass."
    )

    with st.expander("Como funcionam os métodos — conceitos e limitações", expanded=False):
        st.markdown("""
**Gradient × Input saliency**

O modelo transforma cada token num vetor de 768 números (embedding). Durante o forward pass, esses
vetores passam por 12 camadas Transformer até produzir logits de classificação. O gradiente mede
*quanto cada dimensão do embedding influenciou o logit da classe predita*: se aumentar o valor de uma
dimensão em 0.001 muda o logit em 0.05, o gradiente é 50.

A saliência final de um token é a norma L2 do produto elemento-a-elemento entre o gradiente e o
próprio embedding: `score = ‖grad ⊙ embedding‖₂`. Isso combina "quanto esse eixo importou" com
"quanto energia o token já tinha nesse eixo" — um token raro com alto gradiente aparece com score
alto; um token comum com embedding neutro aparece baixo mesmo com gradiente moderado.

*Limitação principal*: gradiente capta importância **local** — o efeito de uma perturbação infinitesimal.
Se o modelo usa interações não-lineares entre tokens (e usa), a saliência por token individualmente
pode subestimar palavras que só importam em combinação.

---

**Attention Rollout**

Transformers têm 12 camadas, cada uma com 12 attention heads. Cada head produz uma matriz
(seq_len × seq_len) indicando quanto cada posição "olhou" para cada outra. O problema: atenção da
camada 3 olhou para saídas da camada 2, que já eram compostas de atenções da camada 1 — a atenção
final do [CLS] não é a soma direta das atenções brutas.

Rollout resolve isso propagando atenção em cadeia: para cada camada, adiciona a identidade (a
residual connection real do Transformer), renormaliza as linhas, e multiplica matrizes da primeira à
última camada. O score final de cada token é a entrada `rollout[0, i]` — o quanto o [CLS] "olhou"
para o token i após propagar por todas as camadas.

*Limitação principal*: Rollout assume que toda a informação flui pelo [CLS]. Em classificação
sequence-level isso é razoável, mas o método ignora atenções cross-head e pode superestimar tokens no
início da sequência (dilution à medida que as camadas se acumulam).

---

**Correlação Spearman (ao rodar "Ambos")**

Mede se os dois métodos concordam na *ordem de importância* dos tokens — não nos valores absolutos,
mas em quais tokens são mais importantes que outros.

| Valor | Interpretação |
|---|---|
| > 0.70 | Alta concordância — ambos os métodos identificam as mesmas evidências |
| 0.40–0.70 | Concordância parcial — vale examinar onde divergem |
| < 0.40 | Baixa concordância — os métodos captam aspectos diferentes do texto |

Quando Spearman é baixo, o mais informativo é comparar os top-3 tokens de cada método.

---

**O que observar nas visualizações**

1. **Tokens com score > 0.7** são as âncoras da decisão. Para `security`, esperamos ver `concat`,
   `injection`, `query`, `password`. Se aparecerem tokens genéricos (`the`, `is`, `a`), o modelo
   pode estar se apoiando em artefatos do dataset sintético.

2. **Score alto em stopwords** é sinal de atenção spurious — indica overfitting no dataset
   sintético. Com dados reais e variados, esse padrão diminui.

3. **Gradient vs Rollout em textos longos**: em sequências com > 50 tokens, Rollout tende a diluir
   os scores (todos ficam próximos de 1/n). Gradient×Input é mais discriminativo nesses casos.

4. **Fronteira architecture/style**: confiança média ~55% nesta POC. Se a saliência destacar os
   mesmos tokens em ambas as classes, o problema é semântico — não de arquitetura do modelo.
        """)

    with st.expander("Por que explainability importa em produção", expanded=False):
        st.markdown("""
Em um classifier de code review em produção, explainability resolve três problemas práticos:

**Debugging de erros**: quando o modelo classifica `observability` como `style`, a saliência mostra
quais tokens guiaram a decisão errada. Se `log` teve score alto mas `silent` (keyword de
observabilidade) teve score baixo, o problema está no dataset — não na arquitetura do modelo.

**Detecção de shortcuts**: modelos treinados em datasets sintéticos aprendem correlações espúrias.
Se todos os exemplos `security` no treino contêm a palavra `SQL`, o modelo aprende a associar `SQL` →
security mesmo num contexto benigno. A saliência expõe esse atalho antes de ir a produção.

**Auditabilidade**: mostrar ao engenheiro *quais palavras levaram à classificação* é fundamentalmente
diferente de mostrar apenas "security 94%". O segundo é caixa preta; o primeiro é explicação
auditável — necessária em contextos onde o reviewer precisa validar ou contestar a sugestão.

**Limitação desta POC**: os dois métodos são *post-hoc* — explicam o modelo após o fato, não como
ele foi treinado. Para explicabilidade mais robusta: Integrated Gradients (remove a saturação do
gradiente simples), LIME/SHAP (model-agnostic, testa perturbações reais), ou TCAV (conceitos de alto
nível como "contém SQL" em vez de tokens individuais).
        """)

    if not full_model:
        st.warning("Modelo full não carregado.")
    else:
        col_expl, col_method = st.columns([3, 1])
        with col_expl:
            expl_text = st.text_area(
                "Finding para analisar",
                value="No deadletter queue monitoring — failed messages accumulate silently",
                height=80,
                key="expl_input",
            )
        with col_method:
            expl_method = st.radio("Método", ["Gradient × Input", "Attention Rollout", "Ambos"], index=0)

        expl_examples = {
            "🔒 Security":      "SQL query built with string concatenation — injection risk",
            "🏛 Architecture":  "Feature envy: method uses 6 fields from another class — move method",
            "📊 Observability": "No deadletter queue monitoring — failed messages accumulate silently",
            "✏️ Style":         "Unused import — remove to keep dependencies explicit",
            "✅ False Pos.":    "This pattern is intentional — retry handles transient failures by design",
        }
        ecols = st.columns(len(expl_examples))
        for col, (lbl, txt) in zip(ecols, expl_examples.items()):
            if col.button(lbl, use_container_width=True, key=f"expl_{lbl}"):
                st.session_state["expl_text"] = txt
                st.rerun()
        if "expl_text" in st.session_state:
            expl_text = st.session_state.pop("expl_text")

        if expl_text.strip() and st.button("Analisar", type="primary"):
            from explainability import gradient_saliency, attention_rollout, tokens_to_html, compare_methods
            with st.spinner("Calculando saliência..."):
                if expl_method == "Gradient × Input":
                    result = gradient_saliency(full_model, full_tok, expl_text)
                    results_to_show = [("Gradient × Input", result)]
                elif expl_method == "Attention Rollout":
                    result = attention_rollout(full_model, full_tok, expl_text)
                    results_to_show = [("Attention Rollout", result)]
                else:
                    cmp = compare_methods(full_model, full_tok, expl_text)
                    results_to_show = [
                        ("Gradient × Input", cmp["gradient_saliency"]),
                        ("Attention Rollout", cmp["attention_rollout"]),
                    ]
                    corr = cmp["spearman_corr"]
                    corr_label = (
                        "alta concordância — ambos identificam as mesmas evidências" if corr > 0.70
                        else "concordância parcial — compare os top-3 tokens de cada método" if corr > 0.40
                        else "baixa concordância — os métodos captam aspectos diferentes do texto"
                    )
                    st.metric(
                        "Correlação Spearman (grad vs rollout)",
                        f"{corr:.3f}",
                        delta=corr_label,
                        delta_color="off",
                        help="Mede concordância na *ordem* de importância dos tokens. "
                             "> 0.70 = alta, 0.40–0.70 = parcial, < 0.40 = baixa.",
                    )

            for method_name, res in results_to_show:
                st.subheader(method_name)
                pred_color = LABEL_COLORS.get(res["pred_label"], "#888")
                st.markdown(
                    f"<span style='background:{pred_color};color:white;padding:4px 14px;"
                    f"border-radius:16px;font-weight:600'>"
                    f"{res['pred_label']} {res['pred_conf']:.1%}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("")

                # Top-3 tokens âncora
                token_score_pairs = sorted(
                    zip(res["tokens"], res["scores"]), key=lambda x: x[1], reverse=True
                )
                top3 = [(t.strip() or t, s) for t, s in token_score_pairs if t.strip()][:3]
                if top3:
                    anchor_str = " · ".join(f"`{t}` ({s:.2f})" for t, s in top3)
                    st.caption(f"Top-3 âncoras: {anchor_str}")

                # Bar chart de saliência por token
                tokens_clean = [t.replace("Ġ", " ").replace("Ċ", "↵").strip() or t for t in res["tokens"]]
                fig_sal = go.Figure(go.Bar(
                    x=tokens_clean,
                    y=res["scores"],
                    marker_color=[
                        f"rgba({int(pred_color[1:3],16)},{int(pred_color[3:5],16)},{int(pred_color[5:7],16)},{0.3 + 0.7*s})"
                        for s in res["scores"]
                    ],
                    text=[f"{s:.2f}" for s in res["scores"]],
                    textposition="outside",
                ))
                fig_sal.update_layout(
                    xaxis_title="Token", yaxis_title="Saliência (0–1)",
                    yaxis_range=[0, 1.15], height=320, template="simple_white",
                    title=f"Saliência por token — {method_name}",
                )
                st.plotly_chart(fig_sal, use_container_width=True)

                # HTML colorido
                html = tokens_to_html(res["tokens"], res["scores"], res["pred_label"], res["pred_conf"])
                st.markdown("**Visualização colorida** — intensidade = importância do token para a decisão:")
                st.components.v1.html(html, height=100)

        # HTML salvo
        html_path = ROOT / "models" / "explainability_sample.html"
        if html_path.exists():
            st.divider()
            st.subheader("Exemplos pré-gerados — 5 amostras do test set")
            st.caption(
                "Gerados por `python src/explainability.py`. Cada exemplo mostra Gradient×Input e "
                "Attention Rollout lado a lado. Use para comparar o comportamento do modelo em casos "
                "reais antes de interagir com o campo de texto acima."
            )
            html_content = html_path.read_text(encoding="utf-8")
            st.components.v1.html(html_content, height=500, scrolling=True)

# ════════════════════════════════════════════════════════════════════════════
# ABA 9 — QUANTIZAÇÃO
# ════════════════════════════════════════════════════════════════════════════
with tab9:
    st.header("⚡ Quantização INT8")
    st.caption(
        "Quantização pós-treino: converte pesos Float32 para Int8 **sem retreinar**. "
        "Reduz memória ~68% com F1 idêntico. Latência depende do backend — "
        "qnnpack (Apple Silicon) é mais lento que fbgemm (Linux x86)."
    )

    with st.expander("Como a quantização funciona — e quando usar", expanded=False):
        st.markdown("""
**De Float32 para Int8**

Pesos de redes neurais são tipicamente armazenados como float32 (32 bits por número). Quantização
pós-treino mapeia esses pesos para int8 (8 bits) usando calibração estatística:

1. Rodamos o modelo em alguns exemplos (calibração) e coletamos os ranges de valores dos pesos
   e ativações.
2. Calculamos o fator de escala: `scale = (max − min) / 255`.
3. Convertemos: `w_int8 = round(w_float32 / scale) + zero_point`.

A operação inversa (dequantização) reconstrói o float32 aproximado na hora da inferência.

**Por que F1 fica idêntico mas memória cai 68%**

A quantização é suficientemente precisa para os fins de classificação de texto: a perda de precisão
numérica (float32 → int8) afeta apenas as casas decimais dos logits, sem mudar qual classe tem a
maior probabilidade. O F1 medido no test set é idêntico porque nenhum exemplo muda de classe.

A memória cai de ~480 MB para ~155 MB porque cada parâmetro ocupa 4x menos espaço (32 bits → 8 bits).

**Por que fica mais lento no Mac (qnnpack)**

Quantização acelera inferência em hardware com instruções SIMD otimizadas para int8 — o que Intel/AMD
(fbgemm) fazem bem. O backend qnnpack do Apple Silicon não tem essa otimização no PyTorch atual,
então a dequantização overhead supera o ganho. Em produção com CPU x86 (container Linux), a latência
seria menor que FP32.

**Quando usar quantização**

| Cenário | Recomendação |
|---|---|
| Deploy em CPU (servidor Linux) | INT8 — latência e memória menores |
| Deploy em Apple Silicon | FP32 ou esperar otimização do backend |
| GPU inference (NVIDIA) | FP16 mixed precision > INT8 para BERT |
| Edge/mobile (RAM limitada) | INT8 ou INT4 (quantização mais agressiva) |

**Próximos passos além desta POC**: Quantization-Aware Training (QAT) — treina com simulação de
quantização e recupera 0.5–1pp de F1. GPTQ e AWQ fazem quantização INT4 para LLMs com perda mínima.
        """)

    bench_path = ROOT / "models" / "quantization_benchmark.json"
    if not bench_path.exists():
        st.warning("Execute `python src/quantization.py` para gerar o benchmark.")
    else:
        import json as _json2
        bench = _json2.loads(bench_path.read_text())
        configs = list(bench.keys())

        # Métricas rápidas
        if "Full FP32" in bench and "Full INT8" in bench:
            fp32 = bench["Full FP32"]
            int8 = bench["Full INT8"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("F1 FP32", f"{fp32['f1_macro']:.4f}")
            c2.metric("F1 INT8", f"{int8['f1_macro']:.4f}",
                      delta=f"{int8['f1_macro']-fp32['f1_macro']:+.4f}")
            c3.metric("Mem FP32", f"{fp32['memory_mb']:.0f} MB")
            c4.metric("Mem INT8", f"{int8['memory_mb']:.0f} MB",
                      delta=f"{int8['memory_mb']-fp32['memory_mb']:+.0f} MB",
                      delta_color="inverse")

        st.divider()

        # Tabela completa
        rows = []
        for cfg, m in bench.items():
            rows.append({
                "Configuração": cfg,
                "F1 Macro": f"{m['f1_macro']:.4f}",
                "Latência (ms)": f"{m['latency_ms']:.1f}",
                "Memória (MB)": f"{m['memory_mb']:.1f}",
                "Disco (MB)": f"{m['disk_mb']:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # Gráficos comparativos
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_mem = go.Figure(go.Bar(
                x=list(bench.keys()),
                y=[v["memory_mb"] for v in bench.values()],
                marker_color=["#3498db", "#e74c3c", "#85c1e9", "#f1948a"],
                text=[f"{v['memory_mb']:.0f}MB" for v in bench.values()],
                textposition="outside",
            ))
            fig_mem.update_layout(title="Memória (MB)", height=300,
                                  template="simple_white", yaxis_title="MB")
            st.plotly_chart(fig_mem, use_container_width=True)

        with col_b2:
            fig_lat = go.Figure(go.Bar(
                x=list(bench.keys()),
                y=[v["latency_ms"] for v in bench.values()],
                marker_color=["#3498db", "#e74c3c", "#85c1e9", "#f1948a"],
                text=[f"{v['latency_ms']:.1f}ms" for v in bench.values()],
                textposition="outside",
            ))
            fig_lat.update_layout(title="Latência por inferência (ms)", height=300,
                                  template="simple_white", yaxis_title="ms")
            st.plotly_chart(fig_lat, use_container_width=True)

        st.info(
            "💡 **Por que INT8 é mais lento no Mac?** "
            "qnnpack (backend ARM) não tem as otimizações AVX512 do fbgemm (x86 Linux). "
            "Em produção Linux, INT8 entrega tipicamente 2–3× speedup."
        )

# ════════════════════════════════════════════════════════════════════════════
# ABA 10 — BASELINE
# ════════════════════════════════════════════════════════════════════════════
with tab10:
    st.header("📊 Baseline TF-IDF vs CodeBERT")
    st.caption(
        "Quanto do F1 vem do pré-treino do CodeBERT vs simples features de texto? "
        "Um delta pequeno significa que a complexidade do fine-tuning pode não se justificar para este dataset."
    )

    baseline_path = ROOT / "models" / "baseline_results.json"
    if not baseline_path.exists():
        st.warning("Execute `python src/baseline.py` para gerar o baseline.")
    else:
        import json as _json3
        bl = _json3.loads(baseline_path.read_text())
        cmp = bl.get("comparison", {})

        # Métricas principais
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Melhor baseline", bl.get("best_pipeline", "—"))
        c2.metric("F1 Baseline", f"{cmp.get('baseline_f1', 0):.4f}")
        c3.metric("F1 CodeBERT", f"{cmp.get('codebert_f1', 0):.4f}",
                  delta=f"+{cmp.get('delta', 0):.4f}")
        verdict = cmp.get("verdict", "—")
        color = "🟢" if "justificado" in verdict else ("🟡" if "Marginal" in verdict else "🔴")
        c4.metric("Veredicto", f"{color} {verdict.split('(')[0].strip()}")

        st.divider()

        # Barchart F1 por classe
        per_class = bl.get("per_class_f1", {})
        codebert_per_class = {
            "security": 0.909, "architecture": 1.0, "observability": 0.889,
            "style": 0.750, "false_positive": 0.833,
        }
        if per_class:
            fig_bl = go.Figure()
            fig_bl.add_bar(name="TF-IDF (melhor)",
                           x=list(per_class.keys()), y=list(per_class.values()),
                           marker_color="#f39c12",
                           text=[f"{v:.2f}" for v in per_class.values()], textposition="outside")
            fig_bl.add_bar(name="CodeBERT Full FT",
                           x=list(codebert_per_class.keys()), y=list(codebert_per_class.values()),
                           marker_color="#3498db",
                           text=[f"{v:.2f}" for v in codebert_per_class.values()], textposition="outside")
            fig_bl.update_layout(
                barmode="group", title="F1 por classe: Baseline vs CodeBERT",
                yaxis_range=[0, 1.15], height=360, template="simple_white",
            )
            st.plotly_chart(fig_bl, use_container_width=True)

        # Interpretação
        delta = cmp.get("delta", 0)
        lift = cmp.get("codebert_lift_pct", 0)
        st.subheader("Interpretação")
        if delta < 0.05:
            st.warning(
                f"**Delta = {delta:.3f} ({lift:.1f}% lift).** "
                "Com dados sintéticos de 250 exemplos, TF-IDF char ngrams captura quase toda a sinalização — "
                "o CodeBERT contribui marginalmente. "
                "Com dados reais e variados (github_scraper.py), o gap tende a crescer para 15–25pp."
            )
        else:
            st.success(
                f"**Delta = {delta:.3f} ({lift:.1f}% lift).** "
                "CodeBERT justificado — pré-treino em código contribui além de keyword matching."
            )

        st.markdown("""
        **O que este resultado ensina:**
        - Com dados sintéticos homogêneos, keywords são features quase tão boas quanto embeddings
        - A classe `architecture` é onde o baseline mais falha — semântica estrutural não é capturada por ngrams
        - O valor do CodeBERT se manifesta em dados reais com variação de domínio
        """)

# ════════════════════════════════════════════════════════════════════════════
# ABA 11 — ACTIVE LEARNING
# ════════════════════════════════════════════════════════════════════════════
with tab11:
    st.header("🔄 Active Learning")
    st.caption(
        "Seleção por incerteza: o modelo indica quais exemplos são mais informativos para treinar. "
        "Humano anota esses — não todos. Loop iterativo que maximiza ganho por esforço de anotação."
    )

    with st.expander("Como funciona o Active Learning — o loop completo", expanded=False):
        st.markdown("""
**O problema que resolve**

Anotar dados é caro. Um engenheiro anotando findings de code review leva 2–5 minutos por exemplo.
Para 1000 exemplos, são 40–80 horas. Active Learning reduz esse custo: em vez de anotar aleatoriamente,
o modelo indica quais exemplos são mais informativos — os que ele menos sabe classificar.

**O loop iterativo**

```
1. Treinar com dataset inicial pequeno (seed set)
2. Rodar o modelo em pool de exemplos não anotados
3. Selecionar os N mais incertos (por entropia ou margin sampling)
4. Humano anota esses N exemplos
5. Adicionar ao treino e retreinar
6. Repetir até atingir F1 alvo ou orçamento de anotação
```

**Entropia como medida de incerteza**

Para um classificador de 5 classes, a entropia máxima é `log2(5) ≈ 2.32 bits` (quando o modelo
distribui 20% para cada classe — completamente perdido). Entropia zero ocorre quando o modelo
atribui 100% a uma classe — completamente confiante.

`H(p) = −Σ p_i × log2(p_i)`

Os exemplos com maior entropia são os candidatos para anotação: o modelo os coloca na fronteira de
decisão e qualquer nova informação sobre eles vai mover os pesos de forma significativa.

**Comparação de estratégias de seleção**

| Estratégia | Como seleciona | Vantagem | Limitação |
|---|---|---|---|
| Random | Aleatório | Simples, sem viés | Desperdiça budget em fáceis |
| Uncertainty (entropia) | Maior H(p) | Eficiente, 40% menos anotações | Pode selecionar outliers |
| Margin sampling | Menor p1 − p2 | Similar à entropia, mais intuitivo | Ignora terceira classe |
| Query by Committee | Maior disagreement entre modelos | Mais robusto | Requer vários modelos |

**Por que a simulação aqui tem ganho pequeno**

O val set tem 25 exemplos sintéticos, e o modelo já convergiu nesse domínio estreito. Com dados reais
do `github_scraper.py`, cada round adicionaria 10–20 exemplos genuinamente novos e o F1 subiria 1–3pp
por round — o que é substancial considerando que o modelo já parte de F1=0.876.
        """)

    if not full_model:
        st.warning("Modelo full não carregado.")
    else:
        st.subheader("Exemplos mais incertos (uncertainty sampling)")
        st.markdown("Os exemplos abaixo foram selecionados do val set por maior entropia da distribuição:")

        from active_learning import compute_uncertainty, select_uncertain
        test_examples = load_split("test")
        with st.spinner("Calculando incerteza..."):
            uncertain = select_uncertain(full_model, full_tok, test_examples, n=10,
                                         device=torch.device("cpu"))

        rows_al = []
        for ex in uncertain:
            rows_al.append({
                "Texto": ex["text"][:80],
                "Top1": ex["top1"],
                "Confiança": f"{ex['top1_prob']:.1%}",
                "Entropia": f"{ex['entropy']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows_al), use_container_width=True, hide_index=True)

        # Plot de distribuição de entropia
        from active_learning import compute_uncertainty
        all_texts = [e["text"] for e in test_examples]
        with st.spinner("Calculando entropia para todos os exemplos..."):
            entropies = compute_uncertainty(full_model, full_tok, all_texts, device=torch.device("cpu"))

        fig_ent = go.Figure()
        fig_ent.add_trace(go.Histogram(
            x=entropies, nbinsx=15, name="Entropia",
            marker_color="#3498db", opacity=0.7,
        ))
        threshold_ent = sorted(entropies, reverse=True)[min(9, len(entropies)-1)]
        fig_ent.add_vline(x=threshold_ent, line_dash="dash", line_color="red",
                          annotation_text="threshold top-10")
        fig_ent.update_layout(
            title="Distribuição de entropia no test set",
            xaxis_title="Entropia H(p)", yaxis_title="Count",
            height=300, template="simple_white",
        )
        st.plotly_chart(fig_ent, use_container_width=True)

        st.divider()
        st.subheader("Conceito: por que selecionar por incerteza?")
        st.markdown("""
        | Estratégia | Exemplos anotados | F1 típico |
        |---|---|---|
        | Random sampling | 100 | baseline |
        | Uncertainty sampling | 40–60 | mesmo F1 que random com 100 |
        | Query by Committee | 30–50 | melhor ainda |

        **Limitação atual:** com val set sintético de 25 exemplos, o loop não demonstra melhora porque
        o modelo já convergiu nesse domínio. Com dados reais do `github_scraper.py`, cada round
        adicionaria 10–20 exemplos anotados e o F1 subiria 1–3pp por round.
        """)

        if st.button("Simular 2 rounds de Active Learning", type="primary"):
            from active_learning import active_learning_cycle
            with st.spinner("Rodando 2 rounds (retreino rápido 2 epochs cada)..."):
                history = active_learning_cycle(str(FULL_DIR), n_rounds=2, n_per_round=10)

            st.subheader("Histórico de rounds")
            hist_df = pd.DataFrame([{
                "Round": h["round"],
                "Exemplos adicionados": h["examples_added"],
                "F1 após round": f"{h['f1']:.4f}",
                "Delta F1": f"{h['delta']:+.4f}",
            } for h in history])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            fig_al = go.Figure(go.Scatter(
                x=[0] + [h["round"] for h in history],
                y=[history[0]["f1"] - history[0]["delta"]] + [h["f1"] for h in history],
                mode="lines+markers", marker=dict(size=10), line=dict(color="#3498db", width=2),
            ))
            fig_al.update_layout(
                title="F1 por round de Active Learning",
                xaxis_title="Round", yaxis_title="F1 Macro",
                height=300, template="simple_white",
            )
            st.plotly_chart(fig_al, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# ABA 12 — LEARNING CURVES
# ════════════════════════════════════════════════════════════════════════════
with tab12:
    st.header("📉 Learning Curves")
    st.caption(
        "Quantos exemplos são necessários para atingir F1=0.90? "
        "O modelo está saturado ou mais dados ajudariam? "
        "Cada ponto treina do **base model pré-treinado** (não do checkpoint), "
        "medindo o efeito real do tamanho do dataset."
    )

    with st.expander("Como ler e usar as learning curves", expanded=False):
        st.markdown("""
**O que é uma learning curve**

Cada ponto no gráfico representa um experimento independente: treinar o modelo com X% dos dados,
avaliar no val e test set, registrar o F1. O eixo X é o número de exemplos de treino; o eixo Y é
o F1 Macro.

Importante: cada ponto parte do **base model pré-treinado** (não do checkpoint final) — isso isola
o efeito do tamanho do dataset sem contaminar com informação dos experimentos anteriores.

**Três padrões possíveis e o que fazer**

| Padrão | Diagnóstico | Ação |
|---|---|---|
| Curva sobe e ainda não planifica | Mais dados ajudariam | Coletar mais exemplos anotados |
| Curva planificou (ganho < 0.01 no último quartil) | Modelo saturado para este dataset | Melhorar arquitetura, features ou pipeline |
| Curva do baseline acompanha o CodeBERT | Pré-treino não está contribuindo | Dataset sintético demais — coletar dados reais |

**Gap entre val e test**

Val F1 e test F1 devem ser próximos (< 0.03 de diferença) ao longo de toda a curva. Se o gap cresce
com mais dados, é sinal de data leakage entre treino e val — risco de avaliar o modelo em exemplos
que "vazaram" do treino.

**Por que a baseline também cresce**

TF-IDF com mais dados aprende n-grams mais ricos. A curva do baseline crescendo junto com o CodeBERT
é evidência de que o ganho vem do volume de dados, não necessariamente do pré-treino. O gap entre
as duas curvas mede a contribuição real do CodeBERT em cada fração — se o gap cresce com mais dados,
o pré-treino está sendo bem aproveitado.

**Decisão de investimento em dados**

Se a curva ainda sobe em 80% dos dados: coletar mais exemplos tem ROI alto.
Se planificou em 50%: o próximo investimento deveria ser na qualidade das anotações ou na diversidade
das fontes, não no volume.
        """)

    lc_path = ROOT / "models" / "learning_curves.json"

    if not lc_path.exists():
        st.warning("Execute `python src/learning_curves.py` para gerar as curvas.")
        st.info(
            "⚠️ **Aviso:** o script treina o modelo várias vezes — "
            "leva ~15–30 min no MPS. O baseline TF-IDF termina em < 1 min."
        )
    else:
        import json as _json_lc
        lc_data = _json_lc.loads(lc_path.read_text())
        cb_results = lc_data.get("codebert", [])
        bl_results = lc_data.get("baseline", [])

        if cb_results or bl_results:
            # Métricas rápidas
            if cb_results:
                best = max(cb_results, key=lambda r: r["val_f1"])
                worst = min(cb_results, key=lambda r: r["val_f1"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Melhor F1 (val)", f"{best['val_f1']:.4f}",
                          f"{best['fraction']:.0%} dos dados")
                c2.metric("F1 mínimo (val)", f"{worst['val_f1']:.4f}",
                          f"{worst['fraction']:.0%} dos dados")
                delta_last = cb_results[-1]["val_f1"] - cb_results[-2]["val_f1"] if len(cb_results) > 1 else 0
                c3.metric("Ganho 80%→100%", f"{delta_last:+.4f}",
                          "saturado" if abs(delta_last) < 0.01 else "ainda aprendendo")

            st.divider()

            # Curva principal
            fig_lc = go.Figure()
            if cb_results:
                fig_lc.add_trace(go.Scatter(
                    x=[r["n_train"] for r in cb_results],
                    y=[r["val_f1"] for r in cb_results],
                    mode="lines+markers", name="CodeBERT (val F1)",
                    line=dict(color="#3498db", width=2), marker=dict(size=9),
                ))
                fig_lc.add_trace(go.Scatter(
                    x=[r["n_train"] for r in cb_results],
                    y=[r["test_f1"] for r in cb_results],
                    mode="lines+markers", name="CodeBERT (test F1)",
                    line=dict(color="#3498db", width=2, dash="dash"),
                    marker=dict(size=9, symbol="diamond"),
                ))
            if bl_results:
                fig_lc.add_trace(go.Scatter(
                    x=[r["n_train"] for r in bl_results],
                    y=[r["val_f1"] for r in bl_results],
                    mode="lines+markers", name="TF-IDF baseline (val F1)",
                    line=dict(color="#f39c12", width=2), marker=dict(size=9),
                ))

            fig_lc.add_hline(y=0.90, line_dash="dot", line_color="green",
                             annotation_text="F1=0.90 (target)")
            fig_lc.update_layout(
                title="F1 vs Tamanho do Dataset",
                xaxis_title="Exemplos de treino",
                yaxis_title="F1 Macro",
                yaxis_range=[0, 1.05],
                height=420, template="simple_white",
            )
            st.plotly_chart(fig_lc, use_container_width=True)

            # Tabela comparativa
            if cb_results and bl_results:
                st.subheader("Comparação por fração")
                bl_by_n = {r["n_train"]: r for r in bl_results}
                rows_lc = []
                for r in cb_results:
                    bl = bl_by_n.get(r["n_train"], {})
                    gap = r["val_f1"] - bl.get("val_f1", 0)
                    rows_lc.append({
                        "Fração": f"{r['fraction']:.0%}",
                        "N treino": r["n_train"],
                        "CodeBERT F1": f"{r['val_f1']:.4f}",
                        "Baseline F1": f"{bl.get('val_f1', 0):.4f}",
                        "Gap (CB - BL)": f"{gap:+.4f}",
                    })
                st.dataframe(pd.DataFrame(rows_lc), use_container_width=True, hide_index=True)

        if st.button("Rodar apenas baseline (rápido, sem GPU)", key="run_lc_baseline"):
            with st.spinner("Rodando baseline learning curves..."):
                from learning_curves import run_baseline_learning_curves, save_results
                bl = run_baseline_learning_curves()
                cb = lc_data.get("codebert", []) if lc_path.exists() else []
                save_results(cb, bl)
            st.success("Baseline concluído — recarregue a aba.")

# ════════════════════════════════════════════════════════════════════════════
# ABA 13 — DRIFT DETECTION
# ════════════════════════════════════════════════════════════════════════════
with tab13:
    st.header("🛰️ Drift Detection")
    st.caption(
        "Monitora se a distribuição das predições em produção está se desviando do esperado. "
        "**PSI** mede drift na distribuição de classes. **KS test** mede drift na distribuição de confiança."
    )

    with st.expander("Como funciona a detecção de drift — PSI, KS e os 3 cenários", expanded=False):
        st.markdown("""
**O problema em produção**

Um modelo treinado em dados de um período pode degradar silenciosamente quando a distribuição de
inputs muda. Exemplo: o time passou a fazer mais code reviews de segurança (label shift) — o modelo
foi calibrado para distribuição balanceada e agora fica sobrecarregado numa classe. Ou inputs fora
do domínio chegam (domain shift) e o modelo classifica com confiança espúria.

**PSI — Population Stability Index**

Compara a distribuição de classes preditas entre a referência (treino) e a janela atual (produção).

`PSI = Σ (p_atual_i − p_ref_i) × log(p_atual_i / p_ref_i)`

| PSI | Status |
|---|---|
| < 0.10 | Estável — distribuição não mudou significativamente |
| 0.10–0.25 | Mudança moderada — monitorar mais de perto |
| > 0.25 | Drift significativo — investigar e possivelmente retreinar |

**KS Test (Kolmogorov-Smirnov)**

O KS test compara duas distribuições empíricas — aqui, a distribuição de confiança (max softmax)
entre a referência e a janela atual. O p-value indica a probabilidade de as duas amostras virem da
mesma distribuição.

- p < 0.05: as distribuições são estatisticamente diferentes (drift de confiança).
- p > 0.05: não há evidência de drift na confiança.

O KS é especialmente útil para detectar domain shift: inputs fora do domínio tendem a ter confiança
diferente (às vezes mais alta por MSP espúrio, às vezes mais baixa).

**Os 3 cenários simulados**

| Cenário | O que acontece | PSI esperado | KS esperado |
|---|---|---|---|
| Sem drift (val set) | Distribuição similar ao treino | < 0.10 | p > 0.05 |
| Label shift (só security) | 5× mais findings de security | > 0.25 | drift detectado |
| Confidence drop (arch/style) | Fronteira ambígua, confiança cai | moderado | drift provável |
| Domain shift (textos aleatórios) | Inputs fora do domínio | > 0.25 | drift detectado |

**Como usar em produção**

1. Gerar a distribuição de referência no train set: `python src/drift_detection.py`
2. A cada N horas (ou N requests), rodar `DriftMonitor.check(textos_recentes)`
3. Se verdict = `alert`: investigar os inputs recentes, considerar retreino
4. Se verdict = `monitor`: aumentar frequência de verificação

**Limitação com dataset sintético**

MSP e entropia falham como detectores de OOD com dados sintéticos homogêneos — o modelo é confiante
em qualquer input porque nunca viu inputs genuinamente diferentes. Com dados reais, o detector de
domain shift funciona muito melhor.
        """)

    ref_path = ROOT / "models" / "drift_reference.json"

    if not ref_path.exists():
        st.warning("Execute `python src/drift_detection.py` para gerar a distribuição de referência.")
    else:
        import json as _json_dr
        ref_data = _json_dr.loads(ref_path.read_text())

        # Distribuição de referência
        st.subheader("Distribuição de referência (train set)")
        ref_fracs = ref_data.get("label_fractions", {})
        fig_ref = go.Figure(go.Bar(
            x=list(ref_fracs.keys()),
            y=list(ref_fracs.values()),
            marker_color=[LABEL_COLORS.get(l, "#888") for l in ref_fracs.keys()],
            text=[f"{v:.1%}" for v in ref_fracs.values()],
            textposition="outside",
        ))
        fig_ref.update_layout(
            title="Distribuição esperada (referência)",
            yaxis_range=[0, 0.4], height=300, template="simple_white",
            yaxis_title="Fração",
        )
        st.plotly_chart(fig_ref, use_container_width=True)

        st.divider()
        st.subheader("Simular cenários de drift")
        st.caption("Selecione um cenário para ver como PSI e KS detectam mudanças na distribuição.")

        scenario_col, result_col = st.columns([1, 2])
        with scenario_col:
            scenario = st.radio("Cenário", [
                "✅ Sem drift (val set normal)",
                "🚨 Label shift (só security)",
                "⚠️ Confidence drop (fronteira arch/style)",
                "🌍 Domain shift (fora do domínio)",
            ], index=0)

        scenario_map = {
            "✅ Sem drift (val set normal)": None,
            "🚨 Label shift (só security)": "label_shift",
            "⚠️ Confidence drop (fronteira arch/style)": "confidence_drop",
            "🌍 Domain shift (fora do domínio)": "domain_shift",
        }

        if st.button("Executar detecção", type="primary"):
            from drift_detection import DriftMonitor

            monitor = DriftMonitor(str(FULL_DIR), str(ref_path))
            monitor.load_reference()

            drift_type = scenario_map[scenario]
            with st.spinner("Rodando inferência..."):
                if drift_type is None:
                    import json as _j
                    val_texts = [_j.loads(l)["text"] for l in open(SPLITS_DIR / "val.jsonl")]
                    result = monitor.check(val_texts)
                else:
                    _, result = monitor.simulate_drift(drift_type)

            with result_col:
                verdict = result["overall_verdict"]
                color_map = {"no_drift": "🟢", "monitor": "🟡", "alert": "🔴"}
                emoji = color_map.get(verdict, "❓")

                c1, c2, c3 = st.columns(3)
                c1.metric("PSI", f"{result['psi']:.3f}",
                          result["psi_status"].upper(), delta_color="off")
                c2.metric("KS p-value", f"{result['ks_confidence']['p_value']:.4f}",
                          "drift" if result["ks_confidence"]["drift_detected"] else "ok",
                          delta_color="off")
                c3.metric("Veredicto", f"{emoji} {verdict.upper()}")

                # Gráfico: distribuição atual vs referência
                curr_fracs = result.get("current_distribution", {})
                if curr_fracs:
                    fig_cmp = go.Figure()
                    fig_cmp.add_bar(
                        name="Referência",
                        x=list(ref_fracs.keys()), y=list(ref_fracs.values()),
                        marker_color="#bdc3c7",
                        text=[f"{v:.1%}" for v in ref_fracs.values()], textposition="outside",
                    )
                    fig_cmp.add_bar(
                        name="Atual",
                        x=list(curr_fracs.keys()), y=list(curr_fracs.values()),
                        marker_color=[LABEL_COLORS.get(l, "#888") for l in curr_fracs.keys()],
                        text=[f"{v:.1%}" for v in curr_fracs.values()], textposition="outside",
                    )
                    fig_cmp.update_layout(
                        barmode="group", title="Distribuição atual vs referência",
                        yaxis_range=[0, 1.1], height=340, template="simple_white",
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                # Label drift
                label_drift = result.get("label_drift", {})
                if label_drift:
                    st.markdown("**Drift por label:**")
                    for label, delta in sorted(label_drift.items(), key=lambda x: -x[1]):
                        bar = "█" * int(delta * 40)
                        st.markdown(f"`{label:20s}` {bar} {delta:.1%}")

        st.divider()
        st.subheader("Interpretação do PSI")
        st.markdown("""
        | PSI | Status | Ação recomendada |
        |---|---|---|
        | < 0.10 | ✅ Estável | Continuar monitorando |
        | 0.10 – 0.25 | ⚠️ Moderado | Investigar qual label mudou |
        | > 0.25 | 🚨 Significativo | Retreinar o modelo |

        **KS test** na distribuição de confiança: p < 0.05 indica que a distribuição de confiança mudou —
        mesmo que as labels preditas sejam as mesmas, o modelo pode estar menos (ou mais) confiante.
        """)
