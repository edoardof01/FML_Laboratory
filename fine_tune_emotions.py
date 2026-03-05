# fine_tune_emotions.py
"""
Baseline Fine-Tuning
====================
Standard DistilBERT fine-tuning on the raw (or cleaned) dataset.
Uses the centralised training pipeline from src.train_utils.
"""
from src.config_utils import get_config
from src.train_utils import run_standard_pipeline

config = get_config()
OUTPUT_DIR = config["output_dir"]

print(f"\n{'='*60}")
print(f"BASELINE FINE-TUNING")
print(f"DATASET: {config['dataset']['name']}")
print(f"OUTPUT:  {OUTPUT_DIR}")
print(f"{'='*60}\n")

trainer, predictions, label_info = run_standard_pipeline(config, OUTPUT_DIR)

print(f"\n{'='*60}\nBASELINE TRAINING COMPLETE\n{'='*60}")