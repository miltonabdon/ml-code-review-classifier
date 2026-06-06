"""
CodeReviewClassifier wraps CodeBERT for 5-class sequence classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["security", "architecture", "observability", "style", "false_positive"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

MODEL_NAME = "microsoft/codebert-base"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model(num_labels: int = len(LABELS), pretrained: bool = True):
    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model = AutoModelForSequenceClassification.from_config(config)
    return model


def load_finetuned(checkpoint_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer
