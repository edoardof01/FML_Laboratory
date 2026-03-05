# MLM_pretraining.py
"""
Masked Language Model (MLM) Domain Adaptation + Supervised Fine-Tuning
======================================================================
Phase 1: Continued pretraining with MLM on the task corpus.
Phase 2: Standard supervised fine-tuning (reuses the centralised pipeline).
"""
import os
import json

from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from src.config_utils import get_config
from src.data_utils import load_and_preprocess_dataset
from src.train_utils import set_seed, run_standard_pipeline

# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================
config = get_config()
DATASET_CONFIG = config["dataset"]
OUTPUT_DIR = config["output_dir"].replace("go-emotions", "continued_pretraining_go-emotions")
TRAIN_CONFIG = config["training"]
MODEL_CONFIG = config["model"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(TRAIN_CONFIG.get("seed", 42))

print(f"\n{'='*60}")
print(f"MLM DOMAIN ADAPTATION + FINE-TUNING")
print(f"DATASET: {DATASET_CONFIG['name']}")
print(f"OUTPUT:  {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ============================================================
# 2. LOAD DATASET (for MLM corpus)
# ============================================================
dataset, label_info = load_and_preprocess_dataset(DATASET_CONFIG)

# ============================================================
# 3. PHASE 1 — CONTINUED PRETRAINING (MLM)
# ============================================================
print("\n--- Phase 1: Continued Pretraining with Masked Language Modeling ---")

mlm_adapted_model_path = os.path.join(OUTPUT_DIR, "mlm_adapted_model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["name"])

if not os.path.exists(mlm_adapted_model_path):
    print("Domain-adapted model not found. Starting pretraining...")

    text_column = DATASET_CONFIG["text_column"]
    corpus = []
    for split in dataset.values():
        corpus.extend(split[text_column])

    print(f"Corpus size: {len(corpus)} documents")
    unlabeled_dataset = Dataset.from_dict({text_column: corpus})

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=MODEL_CONFIG.get("max_length", 128),
            return_special_tokens_mask=True,
        )

    tokenized_unlabeled = unlabeled_dataset.map(
        tokenize_function, batched=True, remove_columns=[text_column]
    )

    mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_CONFIG["name"])

    mlm_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "mlm_results"),
        overwrite_output_dir=True,
        per_device_train_batch_size=64,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=100,
        report_to="tensorboard",
    )

    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
        train_dataset=tokenized_unlabeled,
    )
    mlm_trainer.train()

    os.makedirs(mlm_adapted_model_path, exist_ok=True)
    mlm_trainer.save_model(mlm_adapted_model_path)
    tokenizer.save_pretrained(mlm_adapted_model_path)
    print(f"\n✓ Domain-adapted model saved to: {mlm_adapted_model_path}")
else:
    print(f"✓ Domain-adapted model found at '{mlm_adapted_model_path}'. Skipping pretraining.")

# ============================================================
# 4. PHASE 2 — SUPERVISED FINE-TUNING (centralised pipeline)
# ============================================================
print("\n--- Phase 2: Supervised Fine-Tuning for Classification ---")

trainer, predictions, label_info = run_standard_pipeline(
    config,
    OUTPUT_DIR,
    model_name_or_path=mlm_adapted_model_path,
)

print(f"\n{'='*60}\nMLM + FINE-TUNING COMPLETE\n{'='*60}")

