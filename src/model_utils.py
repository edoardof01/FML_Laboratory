# model_utils.py
"""Model initialisation helpers."""
from transformers import AutoModelForSequenceClassification


def get_model(model_name: str, num_labels: int, is_multilabel: bool, label_names=None):
    """
    Load a pre-trained model for sequence classification.

    Args:
        model_name: HuggingFace model id or local path.
        num_labels: Number of output labels.
        is_multilabel: Whether the task is multi-label.
        label_names: Optional list of human-readable label names.

    Returns:
        AutoModelForSequenceClassification instance.
    """
    problem_type = "multi_label_classification" if is_multilabel else "single_label_classification"

    print(f"Loading Model: {model_name}")
    print(f"  Problem Type: {problem_type}")
    print(f"  Num Labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
    )

    if label_names:
        model.config.id2label = {i: label for i, label in enumerate(label_names)}
        model.config.label2id = {label: i for i, label in enumerate(label_names)}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = trainable_params / total_params * 100 if total_params else 0
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({pct:.0f}%)")

    return model
