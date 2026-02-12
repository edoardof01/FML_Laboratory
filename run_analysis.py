# run_analysis.py
import os
import torch
import json
from transformers import AutoTokenizer
from src.analysis_utils import find_hard_negatives
from src.model_utils import get_model
from src.config_utils import get_config
from src.data_utils import load_and_preprocess_dataset, tokenize_dataset

# Load Config
config = get_config()
DATASET_CONFIG = config["dataset"]
OUTPUT_DIR = config["output_dir"]
MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")

# Load Dataset
dataset, label_info = load_and_preprocess_dataset(DATASET_CONFIG)
test_dataset = dataset["test"]

# Load Model & Tokenizer
print(f"Loading model from {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# We use the standard load for analysis, as we want to test the *saved* model
model = get_model(
    model_name=MODEL_DIR,
    num_labels=label_info["num_labels"],
    is_multilabel=label_info["is_multilabel"],
    label_names=label_info["label_names"]
)

# Run Analysis
hard_negatives = find_hard_negatives(
    model, 
    test_dataset.select(range(100)), # Analyze first 100 for speed demo
    tokenizer
)

print("\n--- Hard Negatives (Confident Failures) ---")
for hn in hard_negatives:
    print(f"\nText: {hn['text']}")
    print(f"Type: {hn['type']}")
    print(f"Confidence: {hn['confidence']:.4f}")
    if "predicted_label" in hn:
        lbl = label_info["label_names"][hn["predicted_label"]]
        print(f"Predicted Class: {lbl}")
