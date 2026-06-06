"""
Explainability methods for CodeReview classifier:
  - Gradient × Input saliency (main)
  - Attention rollout (secondary)
  - HTML visualization
  - Method comparison
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from model import LABELS, ID2LABEL


LABEL_COLORS = {
    "security": "#e74c3c",
    "architecture": "#3498db",
    "observability": "#2ecc71",
    "style": "#f39c12",
    "false_positive": "#9b59b6",
}


def _clean_token(token: str) -> str:
    return token.replace("Ġ", " ").replace("Ċ", "\n")


def _remove_special_tokens(tokens, scores, input_ids, tokenizer):
    """Strip [CLS], [SEP], and PAD tokens, return cleaned lists."""
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    }
    filtered_tokens = []
    filtered_scores = []
    for tok, score, tid in zip(tokens, scores, input_ids):
        if tid not in special_ids:
            filtered_tokens.append(_clean_token(tok))
            filtered_scores.append(float(score))
    return filtered_tokens, filtered_scores


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _get_token_bg(score: float, label: str) -> str:
    color = LABEL_COLORS.get(label, "#cccccc")
    r, g, b = _hex_to_rgb(color)
    score = max(0.0, min(1.0, score))
    bg_r = int(255 + (r - 255) * score)
    bg_g = int(255 + (g - 255) * score)
    bg_b = int(255 + (b - 255) * score)
    return f"rgb({bg_r},{bg_g},{bg_b})"


def _get_text_color(score: float, label: str) -> str:
    color = LABEL_COLORS.get(label, "#cccccc")
    r, g, b = _hex_to_rgb(color)
    score = max(0.0, min(1.0, score))
    bg_r = int(255 + (r - 255) * score)
    bg_g = int(255 + (g - 255) * score)
    bg_b = int(255 + (b - 255) * score)
    brightness = bg_r * 0.299 + bg_g * 0.587 + bg_b * 0.114
    return "#000000" if brightness > 128 else "#ffffff"


def gradient_saliency(
    model, tokenizer, text: str, true_label: int | None = None
) -> dict:
    """
    Gradient × Input saliency.

    Returns:
      tokens: list[str]
      scores: list[float]   — normalized 0-1
      pred_label: str
      pred_conf: float
      true_label: str | None
    """
    # Always on CPU for gradient stability (MPS has known issues with backward)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Compute base embeddings without grad, then detach → leaf tensor
    with torch.no_grad():
        embeddings_base = model.roberta.embeddings(input_ids)

    # Leaf tensor that grad will populate after backward
    embeddings = embeddings_base.detach().requires_grad_(True)

    # Forward with inputs_embeds instead of input_ids
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits  # (1, num_labels)

    probs = F.softmax(logits, dim=-1)
    pred_idx = int(torch.argmax(probs, dim=-1).item())
    pred_conf = float(probs[0, pred_idx].item())
    pred_label = ID2LABEL[pred_idx]

    # Backward on the logit of predicted class (or true_label if provided)
    target_idx = true_label if true_label is not None else pred_idx
    model.zero_grad()
    logits[0, target_idx].backward()

    # Saliency: L2 norm over embedding dim of (grad × embedding)
    grad = embeddings.grad  # (1, seq_len, hidden_size)
    saliency = grad * embeddings.detach()  # element-wise product
    scores_raw = saliency.norm(dim=-1).squeeze(0)  # (seq_len,)
    scores_np = scores_raw.detach().cpu().numpy()

    # Normalize 0-1
    s_min, s_max = scores_np.min(), scores_np.max()
    if s_max - s_min > 1e-9:
        scores_np = (scores_np - s_min) / (s_max - s_min)
    else:
        scores_np = np.zeros_like(scores_np)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    ids_list = input_ids[0].tolist()

    tokens, scores = _remove_special_tokens(
        all_tokens, scores_np.tolist(), ids_list, tokenizer
    )

    return {
        "tokens": tokens,
        "scores": scores,
        "pred_label": pred_label,
        "pred_conf": pred_conf,
        "true_label": ID2LABEL[true_label] if true_label is not None else None,
    }


def attention_rollout(model, tokenizer, text: str) -> dict:
    """
    Attention rollout: propagates attention across all layers via matrix product.

    Returns same format as gradient_saliency.
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Force eager attention so output_attentions=True works (SDPA does not support it)
    original_attn_impl = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
    finally:
        # Restore original implementation
        if original_attn_impl is not None:
            model.config._attn_implementation = original_attn_impl
        else:
            del model.config._attn_implementation

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    pred_idx = int(torch.argmax(probs, dim=-1).item())
    pred_conf = float(probs[0, pred_idx].item())
    pred_label = ID2LABEL[pred_idx]

    # outputs.attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
    attentions = outputs.attentions
    seq_len = input_ids.shape[1]
    identity = torch.eye(seq_len, device=device)

    rollout = identity.clone()
    for attn_layer in attentions:
        # Average over heads → (seq_len, seq_len)
        attn_avg = attn_layer[0].mean(dim=0)
        # Add residual connection and re-normalize rows
        a = 0.5 * attn_avg + 0.5 * identity
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(a, rollout)

    # Score per token = rollout row of [CLS] (index 0)
    scores_raw = rollout[0].cpu().numpy()  # (seq_len,)

    s_min, s_max = scores_raw.min(), scores_raw.max()
    if s_max - s_min > 1e-9:
        scores_raw = (scores_raw - s_min) / (s_max - s_min)
    else:
        scores_raw = np.zeros_like(scores_raw)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    ids_list = input_ids[0].tolist()

    tokens, scores = _remove_special_tokens(
        all_tokens, scores_raw.tolist(), ids_list, tokenizer
    )

    return {
        "tokens": tokens,
        "scores": scores,
        "pred_label": pred_label,
        "pred_conf": pred_conf,
        "true_label": None,
    }


def tokens_to_html(
    tokens: list[str],
    scores: list[float],
    pred_label: str,
    pred_conf: float,
) -> str:
    """
    Generates HTML with tokens colored by saliency.
    Background goes from white (score=0) to label color (score=1).
    Tooltip shows score on hover.
    """
    label_color = LABEL_COLORS.get(pred_label, "#cccccc")
    r, g, b = _hex_to_rgb(label_color)

    spans = []
    for token, score in zip(tokens, scores):
        score = max(0.0, min(1.0, score))
        bg_r = int(255 + (r - 255) * score)
        bg_g = int(255 + (g - 255) * score)
        bg_b = int(255 + (b - 255) * score)
        brightness = bg_r * 0.299 + bg_g * 0.587 + bg_b * 0.114
        text_color = "#000000" if brightness > 128 else "#ffffff"
        bg = f"rgb({bg_r},{bg_g},{bg_b})"
        display = token if token.strip() else "&nbsp;"
        span = (
            f'<span style="background:{bg};color:{text_color};padding:2px 1px;'
            f'border-radius:2px;margin:1px;display:inline-block;" '
            f'title="score: {score:.4f}">{display}</span>'
        )
        spans.append(span)

    body = " ".join(spans)
    label_badge = (
        f'<span style="background:{label_color};color:#fff;padding:3px 8px;'
        f'border-radius:4px;font-weight:bold;">{pred_label}</span>'
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Explainability — {pred_label}</title>
  <style>
    body {{ font-family: monospace; padding: 20px; background: #fafafa; }}
    .header {{ margin-bottom: 12px; }}
    .tokens {{ line-height: 2.2; word-spacing: 2px; }}
  </style>
</head>
<body>
  <div class="header">
    <strong>Predicted:</strong> {label_badge} &nbsp;
    <strong>Confidence:</strong> {pred_conf:.2%}
  </div>
  <div class="tokens">{body}</div>
</body>
</html>"""
    return html


def compare_methods(model, tokenizer, text: str) -> dict:
    """
    Runs gradient_saliency and attention_rollout on the same text.
    Returns both results + Spearman correlation between their scores.
    """
    grad_result = gradient_saliency(model, tokenizer, text)
    roll_result = attention_rollout(model, tokenizer, text)

    # Align by shortest token list (tokenizer is deterministic, lengths should match)
    min_len = min(len(grad_result["scores"]), len(roll_result["scores"]))
    g_scores = grad_result["scores"][:min_len]
    r_scores = roll_result["scores"][:min_len]

    if min_len > 1:
        corr, pvalue = spearmanr(g_scores, r_scores)
    else:
        corr, pvalue = float("nan"), float("nan")

    return {
        "gradient_saliency": grad_result,
        "attention_rollout": roll_result,
        "spearman_corr": float(corr),
        "spearman_pvalue": float(pvalue),
    }


def _print_ascii(result: dict, method: str):
    tokens = result["tokens"]
    scores = result["scores"]
    pred = result["pred_label"]
    conf = result["pred_conf"]
    true_lbl = result.get("true_label")

    label_str = f"pred={pred} ({conf:.2%})"
    if true_lbl:
        label_str += f"  true={true_lbl}"
    print(f"\n[{method}] {label_str}")
    for tok, sc in zip(tokens, scores):
        bar = "#" * int(sc * 20)
        print(f"  {tok:20s}  [{sc:.4f}]  {bar}")


if __name__ == "__main__":
    from model import load_finetuned, LABEL2ID

    CHECKPOINT = str(Path(__file__).parent.parent / "models" / "full_finetuned")
    TEST_FILE = Path(__file__).parent.parent / "data" / "splits" / "test.jsonl"
    OUT_HTML = Path(__file__).parent.parent / "models" / "explainability_sample.html"

    print(f"Loading model from {CHECKPOINT} ...")
    model, tokenizer = load_finetuned(CHECKPOINT)
    model.eval()

    examples = []
    with open(TEST_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if len(examples) >= 5:
                break

    all_html_sections = []

    for i, ex in enumerate(examples):
        text = ex["text"]
        true_lbl_str = ex.get("label")
        true_lbl_id = LABEL2ID.get(true_lbl_str) if true_lbl_str else None

        print(f"\n{'='*60}")
        print(f"Example {i+1}: {text[:80]}")

        g_result = gradient_saliency(model, tokenizer, text, true_label=true_lbl_id)
        _print_ascii(g_result, "gradient×input")

        r_result = attention_rollout(model, tokenizer, text)
        _print_ascii(r_result, "attention_rollout")

        min_len = min(len(g_result["scores"]), len(r_result["scores"]))
        if min_len > 1:
            g_s = g_result["scores"][:min_len]
            r_s = r_result["scores"][:min_len]
            # Guard against constant arrays (spearman undefined)
            if np.std(g_s) > 1e-9 and np.std(r_s) > 1e-9:
                corr, _ = spearmanr(g_s, r_s)
                print(f"\n  Spearman(grad, rollout) = {corr:.4f}")
            else:
                print("\n  Spearman: undefined (constant scores in one method)")

        # HTML section
        true_label_display = f" | true: {true_lbl_str}" if true_lbl_str else ""
        section_header = (
            f'<h2 style="margin-top:40px;font-family:monospace;">'
            f'Example {i+1}{true_label_display}</h2>'
            f'<p style="font-family:monospace;color:#555;">{text[:120]}</p>'
        )

        grad_html_inner = " ".join(
            f'<span style="background:{_get_token_bg(sc, g_result["pred_label"])};'
            f'color:{_get_text_color(sc, g_result["pred_label"])};'
            f'padding:2px 1px;border-radius:2px;margin:1px;display:inline-block;" '
            f'title="score:{sc:.4f}">{tok if tok.strip() else "&nbsp;"}</span>'
            for tok, sc in zip(g_result["tokens"], g_result["scores"])
        )
        roll_html_inner = " ".join(
            f'<span style="background:{_get_token_bg(sc, r_result["pred_label"])};'
            f'color:{_get_text_color(sc, r_result["pred_label"])};'
            f'padding:2px 1px;border-radius:2px;margin:1px;display:inline-block;" '
            f'title="score:{sc:.4f}">{tok if tok.strip() else "&nbsp;"}</span>'
            for tok, sc in zip(r_result["tokens"], r_result["scores"])
        )

        label_color_g = LABEL_COLORS.get(g_result["pred_label"], "#ccc")
        label_color_r = LABEL_COLORS.get(r_result["pred_label"], "#ccc")

        section = (
            section_header
            + f'<div style="margin-bottom:8px"><strong>Gradient×Input</strong> — '
            + f'<span style="background:{label_color_g};color:#fff;padding:2px 8px;'
            + f'border-radius:3px">{g_result["pred_label"]}</span> '
            + f'{g_result["pred_conf"]:.2%}</div>'
            + f'<div style="font-family:monospace;line-height:2.4;">{grad_html_inner}</div>'
            + f'<div style="margin-top:16px;margin-bottom:8px"><strong>Attention Rollout</strong> — '
            + f'<span style="background:{label_color_r};color:#fff;padding:2px 8px;'
            + f'border-radius:3px">{r_result["pred_label"]}</span> '
            + f'{r_result["pred_conf"]:.2%}</div>'
            + f'<div style="font-family:monospace;line-height:2.4;">{roll_html_inner}</div>'
        )
        all_html_sections.append(section)

    full_html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Explainability — CodeReview Classifier</title>
  <style>
    body { padding: 30px; background: #f8f8f8; font-size: 14px; }
    h1 { font-family: sans-serif; }
    hr { border: none; border-top: 1px solid #ddd; margin: 30px 0; }
  </style>
</head>
<body>
  <h1>CodeReview Classifier — Explainability Sample</h1>
  <p>5 examples from test set. Methods: Gradient&times;Input saliency and Attention Rollout.</p>
""" + "<hr>".join(all_html_sections) + """
</body>
</html>"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(full_html, encoding="utf-8")
    print(f"\nHTML saved to {OUT_HTML}")
