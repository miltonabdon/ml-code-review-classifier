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


# ─── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 ML Code Review")
    st.caption("CodeBERT · Fine-tuning · LoRA")
    st.divider()

    full_model, full_tok = load_full_model()
    lora_model, lora_tok = load_lora_model()

    st.markdown("**Status dos modelos**")
    st.markdown(f"{'✅' if full_model else '❌'} Full Fine-tuning")
    st.markdown(f"{'✅' if lora_model else '❌'} LoRA Adapter")

    if SPLITS_DIR.exists():
        st.markdown("**Dataset**")
        for split in ["train", "val", "test"]:
            p = SPLITS_DIR / f"{split}.jsonl"
            if p.exists():
                n = sum(1 for _ in open(p))
                st.markdown(f"· {split}: {n} exemplos")

# ─── abas ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "📈 Experimentos", "⚖️ Comparação", "🎯 Inferência", "🧪 Simulador",
    "🎯 Calibração", "🚨 OOD", "🔬 Erros",
    "🧠 Explainability", "⚡ Quantização", "📊 Baseline", "🔄 Active Learning",
])

# ════════════════════════════════════════════════════════════════════════════
# ABA 1 — EXPERIMENTOS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Experimentos MLflow")

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
        template="plotly_white",
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
        template="plotly_white",
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
            template="plotly_white",
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
                    template="plotly_white",
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
            template="plotly_white",
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
            template="plotly_white",
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
            barmode="group", height=380, template="plotly_white",
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
                    height=400, template="plotly_white",
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
                yaxis_range=[0, 1.1], height=320, template="plotly_white", showlegend=False,
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
            yaxis_range=[0, 1.1], height=320, template="plotly_white",
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
        "**Gradient × Input saliency**: para cada token, saliência = ||grad × embedding||₂. "
        "Tokens com score alto foram os mais determinantes para a predição. "
        "**Attention Rollout**: propaga atenção através de todas as camadas via produto matricial."
    )

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
                        ("Gradient × Input", cmp["gradient"]),
                        ("Attention Rollout", cmp["rollout"]),
                    ]
                    st.metric("Correlação Spearman (grad vs rollout)", f"{cmp['spearman']:.3f}",
                              help="1=idênticos, 0=sem correlação, -1=opostos")

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
                    yaxis_range=[0, 1.15], height=320, template="plotly_white",
                    title=f"Saliência por token — {method_name}",
                )
                st.plotly_chart(fig_sal, use_container_width=True)

                # HTML colorido
                html = tokens_to_html(res["tokens"], res["scores"], res["pred_label"], res["pred_conf"])
                st.markdown("**Visualização colorida:**")
                st.components.v1.html(html, height=100)

        # HTML salvo
        html_path = ROOT / "models" / "explainability_sample.html"
        if html_path.exists():
            st.divider()
            st.subheader("Exemplos pré-gerados (5 exemplos do test set)")
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
                                  template="plotly_white", yaxis_title="MB")
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
                                  template="plotly_white", yaxis_title="ms")
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
                yaxis_range=[0, 1.15], height=360, template="plotly_white",
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
            height=300, template="plotly_white",
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
                height=300, template="plotly_white",
            )
            st.plotly_chart(fig_al, use_container_width=True)
