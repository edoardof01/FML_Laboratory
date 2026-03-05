# data_utils.py
"""Dataset loading, label detection, binary conversion, and tokenization."""
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


def load_and_preprocess_dataset(config: dict):
    """
    Load dataset based on configuration and extract label info.

    Returns:
        (dataset, label_info) where label_info is a dict with
        label_names, num_labels, is_multilabel.
    """
    print("--- Loading Dataset ---")
    if config.get("path"):
        print(f"Loading from disk: {config['path']}")
        dataset = load_from_disk(config["path"])
    elif config.get("subset"):
        dataset = load_dataset(config["name"], config["subset"])
    else:
        dataset = load_dataset(config["name"])

    label_column = config.get("label_column", "labels")

    # Extract label info from the training split
    train_split = dataset[config["split_names"]["train"]]
    label_info = extract_label_info(train_split, label_column)

    print(f"✓ Loaded {config['name']}")
    if label_info["is_multilabel"]:
        print(f"✓ Multi-label classification: {label_info['num_labels']} classes")
        dataset = convert_dataset_to_binary(dataset, label_column, label_info["num_labels"])
    else:
        print(f"✓ Single-label classification: {label_info['num_labels']} classes")

    # Normalise label column name
    if label_column != "labels":
        dataset = dataset.rename_column(label_column, "labels")

    return dataset, label_info


def extract_label_info(dataset_split, label_column: str) -> dict:
    """Determine single-label / multi-label and get label names."""
    info = {"label_names": [], "num_labels": 0, "is_multilabel": False}

    try:
        features = dataset_split.features[label_column]
        if hasattr(features, "names"):
            # Single-label (ClassLabel)
            info["label_names"] = features.names
            info["num_labels"] = len(info["label_names"])
            info["is_multilabel"] = False
            return info
        if hasattr(features, "feature") and hasattr(features.feature, "names"):
            # Multi-label (Sequence of ClassLabel)
            info["label_names"] = features.feature.names
            info["num_labels"] = len(info["label_names"])
            info["is_multilabel"] = True
            return info
    except Exception:
        pass

    # Fallback: inspect first sample
    sample = dataset_split[0][label_column]
    if isinstance(sample, (list, tuple, np.ndarray)):
        info["is_multilabel"] = True
        max_idx = 0
        for i in range(min(1000, len(dataset_split))):
            lbl = dataset_split[i][label_column]
            if isinstance(lbl, list) and lbl:
                max_idx = max(max_idx, max(lbl))
        info["num_labels"] = max_idx + 1
        info["label_names"] = [f"LABEL_{i}" for i in range(info["num_labels"])]
    else:
        info["is_multilabel"] = False

    return info


def convert_dataset_to_binary(dataset, label_column: str, num_labels: int):
    """Convert multi-label indices to binary vectors."""

    def _convert(example):
        raw = example[label_column]
        # Already binary?
        if len(raw) == num_labels and all(x in (0, 1) for x in raw):
            return {label_column: [float(x) for x in raw]}
        binary = [0.0] * num_labels
        for idx in raw:
            if 0 <= idx < num_labels:
                binary[idx] = 1.0
        return {label_column: binary}

    print("--- Converting Multi-label to Binary Vectors ---")
    return dataset.map(_convert)


def tokenize_dataset(
    dataset,
    text_column: str = "text",
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
):
    """Tokenize the dataset and set torch format.

    Returns:
        (tokenized_datasets, tokenizer)
    """
    print("\n--- Tokenizing Dataset ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_function, batched=True)

    columns_to_keep = {"input_ids", "attention_mask", "labels"}
    columns_to_remove = [c for c in tokenized["train"].column_names if c not in columns_to_keep]
    tokenized = tokenized.remove_columns(columns_to_remove)
    tokenized.set_format("torch")

    print("✓ Tokenization complete")
    return tokenized, tokenizer
