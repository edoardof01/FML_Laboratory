# fine_tune_clean.py
"""
Fine-Tuning on the Cleaned Dataset (Confident Learning)
========================================================
Same pipeline as fine_tune_emotions.py but overrides the dataset
source to point at the cleaned version produced by clean_dataset.py.
"""
import copy
import os

from src.config_utils import get_config
from src.train_utils import run_standard_pipeline

config = get_config()

# Override dataset source to point at the cleaned version
dataset_cfg = copy.deepcopy(config["dataset"])
dataset_cfg["path"] = "data/go_emotions_cleaned"
dataset_cfg.pop("subset", None)  # remove remote-only key so load_from_disk is used
config["dataset"] = dataset_cfg

DATASET_NAME = dataset_cfg["name"].replace("_", "-")
OUTPUT_DIR = f"./outputs/clean_baseline_{DATASET_NAME}"

print(f"\n{'='*60}")
print(f"CLEAN-DATASET BASELINE")
print(f"Dataset (cleaned): {dataset_cfg['path']}")
print(f"OUTPUT: {OUTPUT_DIR}")
print(f"{'='*60}\n")

trainer, predictions, label_info = run_standard_pipeline(config, OUTPUT_DIR)

print(f"\n{'='*60}\nCLEAN-BASELINE TRAINING COMPLETE\n{'='*60}")
