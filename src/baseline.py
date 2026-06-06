"""
Sklearn baseline pipeline for comparison with CodeBERT fine-tuned model.

Three pipelines evaluated on val set, best selected by F1 macro:
  1. TF-IDF word unigrams+bigrams -> LogisticRegression
  2. TF-IDF char 3-5grams -> LogisticRegression
  3. FeatureUnion (word + char TF-IDF) -> LinearSVC
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))

MODELS_DIR = Path(__file__).parent.parent / "models"
SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"


def _load_jsonl(path: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def _build_pipelines() -> list[tuple[str, Pipeline]]:
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )
    union_tfidf = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=40_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=20_000,
            sublinear_tf=True,
            min_df=2,
        )),
    ])

    return [
        (
            "word_lr",
            Pipeline([
                ("tfidf", word_tfidf),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ]),
        ),
        (
            "char_lr",
            Pipeline([
                ("tfidf", char_tfidf),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ]),
        ),
        (
            "union_svc",
            Pipeline([
                ("features", union_tfidf),
                ("clf", LinearSVC(max_iter=2000, C=1.0, random_state=42)),
            ]),
        ),
    ]


def train_baseline(train_path: str, val_path: str) -> Pipeline:
    train_texts, train_labels = _load_jsonl(train_path)
    val_texts, val_labels = _load_jsonl(val_path)

    pipelines = _build_pipelines()
    best_pipeline = None
    best_f1 = -1.0

    for name, pipeline in pipelines:
        cv_scores = cross_val_score(
            pipeline, train_texts, train_labels,
            cv=3, scoring="f1_macro", n_jobs=-1,
        )
        print(f"  [{name}] CV F1 macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipeline.fit(train_texts, train_labels)
        val_preds = pipeline.predict(val_texts)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"  [{name}] Val F1 macro: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_pipeline = pipeline
            best_name = name

    print(f"\nMelhor pipeline: {best_name} (val F1={best_f1:.4f})")
    return best_pipeline


def evaluate_baseline(pipeline: Pipeline, test_path: str) -> dict:
    test_texts, test_labels = _load_jsonl(test_path)
    preds = pipeline.predict(test_texts)

    f1_macro = f1_score(test_labels, preds, average="macro")
    f1_weighted = f1_score(test_labels, preds, average="weighted")
    report = classification_report(test_labels, preds)

    labels_sorted = sorted(set(test_labels))
    f1_per_label = f1_score(test_labels, preds, average=None, labels=labels_sorted)
    per_class_f1 = {label: float(f1) for label, f1 in zip(labels_sorted, f1_per_label)}

    return {
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "report": report,
        "per_class_f1": per_class_f1,
    }


def compare_with_codebert(baseline_results: dict, codebert_f1: float) -> dict:
    baseline_f1 = baseline_results["f1_macro"]
    delta = codebert_f1 - baseline_f1

    if baseline_f1 > 0:
        lift_pct = (delta / baseline_f1) * 100
    else:
        lift_pct = float("inf")

    if delta > 0.10:
        verdict = "CodeBERT justificado (delta > 10pp)"
    elif delta > 0.03:
        verdict = "Marginal"
    else:
        verdict = "Baseline suficiente"

    return {
        "baseline_f1": round(baseline_f1, 4),
        "codebert_f1": round(codebert_f1, 4),
        "delta": round(delta, 4),
        "codebert_lift_pct": round(lift_pct, 2),
        "verdict": verdict,
    }


def save_baseline(pipeline: Pipeline, path: str = "models/baseline") -> None:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "pipeline.joblib")
    print(f"Pipeline salvo em: {out / 'pipeline.joblib'}")


if __name__ == "__main__":
    import json as _json

    train_path = str(SPLITS_DIR / "train.jsonl")
    val_path = str(SPLITS_DIR / "val.jsonl")
    test_path = str(SPLITS_DIR / "test.jsonl")
    baseline_dir = MODELS_DIR / "baseline"

    print("=== Treinando baseline ===")
    pipeline = train_baseline(train_path, val_path)

    print("\n=== Avaliando no test set ===")
    results = evaluate_baseline(pipeline, test_path)
    print(f"F1 macro:    {results['f1_macro']:.4f}")
    print(f"F1 weighted: {results['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(results["report"])
    print("Per-class F1:")
    for label, f1 in sorted(results["per_class_f1"].items()):
        print(f"  {label}: {f1:.4f}")

    CODEBERT_F1 = 0.876
    print(f"\n=== Comparação com CodeBERT (F1={CODEBERT_F1}) ===")
    comparison = compare_with_codebert(results, CODEBERT_F1)
    for k, v in comparison.items():
        print(f"  {k}: {v}")

    save_baseline(pipeline, str(baseline_dir))

    results_path = MODELS_DIR / "baseline_results.json"
    payload = {**results, "comparison": comparison}
    payload.pop("report")  # não serializar texto longo no JSON raiz
    payload["report"] = results["report"]
    with open(results_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {results_path}")
