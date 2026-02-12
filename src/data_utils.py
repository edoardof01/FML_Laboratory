#data_utils.py
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import numpy as np

def load_and_preprocess_dataset(config):
    """
    Load dataset based on configuration and extract label info.
    """
    print("--- Loading Dataset ---")
    if "path" in config and config["path"]:
        print(f"Loading from disk: {config['path']}")
        dataset = load_from_disk(config["path"])
    elif "subset" in config and config["subset"]:
        dataset = load_dataset(config["name"], config["subset"])
    else:
        dataset = load_dataset(config["name"])
    
    label_column = config.get("label_column", "labels")
    text_column = config.get("text_column", "text")
    
    # Extract label info
    train_split = dataset[config["split_names"]["train"]]
    label_info = extract_label_info(train_split, label_column)
    
    print(f"✓ Loaded {config['name']}")
    if label_info["is_multilabel"]:
        print(f"✓ Multi-label classification: {label_info['num_labels']} classes")
        # Convert to binary vectors if needed
        dataset = convert_dataset_to_binary(dataset, label_column, label_info["num_labels"])
    else:
        print(f"✓ Single-label classification: {label_info['num_labels']} classes")
        
    # Rename label column to 'labels' if needed
    if label_column != "labels":
        dataset = dataset.rename_column(label_column, "labels")
        
    return dataset, label_info

def extract_label_info(dataset_split, label_column):
    """
    Determine if dataset is single-label or multi-label and get label names.
    """
    info = {
        "label_names": [],
        "num_labels": 0,
        "is_multilabel": False
    }
    
    # Try using dataset features first
    try:
        features = dataset_split.features[label_column]
        if hasattr(features, "names"):
            # Single-label
            info["label_names"] = features.names
            info["num_labels"] = len(info["label_names"])
            info["is_multilabel"] = False
        elif hasattr(features, "feature") and hasattr(features.feature, "names"):
            # Multi-label
            info["label_names"] = features.feature.names
            info["num_labels"] = len(info["label_names"])
            info["is_multilabel"] = True
            return info
    except Exception:
        pass
        
    # Fallback: inspection (simplified from weighted_emotions.py logic)
    sample = dataset_split[0][label_column]
    if isinstance(sample, (list, tuple, np.ndarray)):
        # Heuristic: find max index to determine num_labels if names not available
        info["is_multilabel"] = True
        # Logic to be refined based on specific dataset needs if features fail
        # For now, assume we can get names or at least count
        if info["num_labels"] == 0:
             # Basic fallback logic to scan a few items
             max_idx = 0
             for i in range(min(1000, len(dataset_split))):
                 lbl = dataset_split[i][label_column]
                 if isinstance(lbl, list) and lbl:
                     max_idx = max(max_idx, max(lbl))
             info["num_labels"] = max_idx + 1
             info["label_names"] = [f"LABEL_{i}" for i in range(info["num_labels"])]
    else:
        # Likely single-label
        info["is_multilabel"] = False
        
    return info

def convert_dataset_to_binary(dataset, label_column, num_labels):
    """
    Convert multi-label indices to binary vectors.
    """
    def _convert(example):
        raw = example[label_column]
        binary = [0.0] * num_labels
        
        # Check if already binary
        if len(raw) == num_labels and all(x in [0, 1] for x in raw):
             return {label_column: [float(x) for x in raw]}

        # Convert indices to binary
        for idx in raw:
            if 0 <= idx < num_labels:
                binary[idx] = 1.0
        return {label_column: binary}

    print("--- Converting Multi-label to Binary Vectors ---")
    return dataset.map(_convert)

def tokenize_dataset(dataset, text_column="text", model_name="distilbert-base-uncased", max_length=128):
    """
    Tokenize the dataset.
    """
    print("\n--- Tokenizing Dataset ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)
    
    tokenized = dataset.map(tokenize_function, batched=True)
    
    # Set format to torch
    columns_to_keep = ['input_ids', 'attention_mask', 'labels']
    columns_to_remove = [c for c in tokenized["train"].column_names if c not in columns_to_keep]
    tokenized = tokenized.remove_columns(columns_to_remove)
    tokenized.set_format("torch")
    
    print("✓ Tokenization complete")
    return tokenized, tokenizer
