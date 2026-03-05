# k-fold-cross-validation.py
"""
K-Fold Cross-Validation
========================
Evaluates model stability via K-Fold CV, then trains a final model
on the original train split for fair comparison.
"""
import os
import json

import numpy as np
import torch
from datasets import concatenate_datasets
from transformers import (
    TrainingArguments,
    Trainer,
    IntervalStrategy,
)
from sklearn.model_selection import KFold

from src.config_utils import get_config
from src.data_utils import load_and_preprocess_dataset, tokenize_dataset
from src.metrics import make_compute_metrics
from src.model_utils import get_model
from src.train_utils import get_data_collator, set_seed, evaluate_and_save
from src.viz_utils import visualize_embeddings


# ============================================================
# CONFIGURATION - Load from centralized config
# ============================================================
config = get_config()
DATASET_CONFIG = config["dataset"]
MODEL_CONFIG = config["model"]

OUTPUT_DIR = f"./kfold_model_{DATASET_CONFIG['name'].replace('_', '-')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"K-FOLD CROSS-VALIDATION")
print(f"DATASET: {DATASET_CONFIG['name']}")
if "path" in DATASET_CONFIG and DATASET_CONFIG.get("path"):
    print(f"  Using cleaned dataset from: {DATASET_CONFIG['path']}")
print(f"OUTPUT: {OUTPUT_DIR}")
print(f"{'='*60}\n")


# ============================================================
# 1. LOAD AND PREPROCESS DATASET (USING CENTRALIZED UTILS)
# ============================================================
# This now supports loading from cleaned dataset if path is specified in config
dataset, label_info = load_and_preprocess_dataset(DATASET_CONFIG)
label_names = label_info["label_names"]
num_labels = label_info["num_labels"]
is_multilabel = label_info["is_multilabel"]

# Combine train and validation
full_train_validation_dataset = concatenate_datasets([
    dataset[DATASET_CONFIG["split_names"]["train"]], 
    dataset[DATASET_CONFIG["split_names"]["validation"]]
])
test_dataset = dataset[DATASET_CONFIG["split_names"]["test"]]

print(f"  Combined Train+Val: {len(full_train_validation_dataset)} samples")
print(f"  Test: {len(test_dataset)} samples")


# ============================================================
# 2. TOKENIZATION
# ============================================================
tokenized_full, tokenizer = tokenize_dataset(
    full_train_validation_dataset,
    text_column=DATASET_CONFIG["text_column"],
    model_name=MODEL_CONFIG["name"],
    max_length=MODEL_CONFIG.get("max_length", 128),
)
# tokenize_dataset wraps a DatasetDict; we combined into a single Dataset,
# so we do it manually for the combined and test sets.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["name"])

def tokenize_function(examples):
    return tokenizer(
        examples[DATASET_CONFIG["text_column"]],
        truncation=True,
        max_length=MODEL_CONFIG.get("max_length", 128),
    )

tokenized_full_train_validation_dataset = full_train_validation_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

for ds_ref in (tokenized_full_train_validation_dataset, tokenized_test_dataset):
    cols_rm = [c for c in ds_ref.column_names if c not in ("input_ids", "attention_mask", "labels")]
    ds_ref = ds_ref.remove_columns(cols_rm)

# re-assign after column removal (map returns new objects)
tokenized_full_train_validation_dataset = tokenized_full_train_validation_dataset.remove_columns(
    [c for c in tokenized_full_train_validation_dataset.column_names if c not in ("input_ids", "attention_mask", "labels")]
)
tokenized_test_dataset = tokenized_test_dataset.remove_columns(
    [c for c in tokenized_test_dataset.column_names if c not in ("input_ids", "attention_mask", "labels")]
)
tokenized_full_train_validation_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch")
print("✓ Tokenization complete")

# ============================================================
# 3. COLLATOR, METRICS, MODEL INIT (centralised)
# ============================================================
data_collator = get_data_collator(tokenizer, is_multilabel, MODEL_CONFIG.get("max_length", 128))
compute_metrics_fn = make_compute_metrics(is_multilabel)

def model_init(trial=None):
    return get_model(
        model_name=MODEL_CONFIG["name"],
        num_labels=num_labels,
        is_multilabel=is_multilabel,
        label_names=label_names,
    )


# ============================================================
# 8. LOAD BEST HYPERPARAMETERS
# ============================================================
print("\n--- Loading Best Hyperparameters ---")

# Search in multiple candidate locations (baseline output dir first)
TRAIN_CONFIG = config.get("training", {})
BASELINE_OUTPUT_DIR = config.get("output_dir", "./outputs/go-emotions")
hp_search_paths = [
    os.path.join(BASELINE_OUTPUT_DIR, "best_hyperparameters.json"),
    f"./optuna_model_{DATASET_CONFIG['name'].replace('_', '-')}/best_hyperparameters.json",
]

best_hyperparameters = {
    'learning_rate': float(TRAIN_CONFIG.get('learning_rate', 2e-5)),
    'per_device_train_batch_size': TRAIN_CONFIG.get('per_device_train_batch_size', 16),
    'num_train_epochs': TRAIN_CONFIG.get('num_train_epochs', 3),
}

hp_loaded = False
for hp_path in hp_search_paths:
    if os.path.exists(hp_path):
        try:
            with open(hp_path, 'r') as f:
                best_hyperparameters = json.load(f)
            print(f"✓ Hyperparameters loaded from: {hp_path}")
            print(f"  {best_hyperparameters}")
            hp_loaded = True
            break
        except Exception as e:
            print(f"✗ Error loading {hp_path}: {e}")

if not hp_loaded:
    print(f"✗ No hyperparameter file found in: {hp_search_paths}")
    print(f"  Using config defaults: {best_hyperparameters}")


# ============================================================
# 4. LOAD BEST HYPERPARAMETERS
# ============================================================
weight_decay = TRAIN_CONFIG.get('weight_decay', 0.01)
seed = TRAIN_CONFIG.get('seed', 42)

training_args_kfold = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "kfold_results"),
    learning_rate=best_hyperparameters.get('learning_rate', 2e-5),
    per_device_train_batch_size=best_hyperparameters.get('per_device_train_batch_size', 16),
    per_device_eval_batch_size=best_hyperparameters.get('per_device_train_batch_size', 16),
    num_train_epochs=best_hyperparameters.get('num_train_epochs', 3),
    weight_decay=weight_decay,
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_dir=os.path.join(OUTPUT_DIR, "kfold_logs"),
    logging_steps=100,
    report_to="tensorboard",
    seed=seed,
)


# ============================================================
# 11. K-FOLD CROSS-VALIDATION
# ============================================================
n_splits = 4
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_fold_metrics = []
indices = np.arange(len(tokenized_full_train_validation_dataset))

print(f"\n--- Starting K-Fold Cross-Validation (K={n_splits}) ---")

for fold, (train_index, val_index) in enumerate(kf.split(indices)):
    print(f"\n***** Fold {fold + 1}/{n_splits} *****")
    
    train_fold_dataset = tokenized_full_train_validation_dataset.select(train_index)
    val_fold_dataset = tokenized_full_train_validation_dataset.select(val_index)
    
    trainer = Trainer(
        model_init=model_init,
        args=training_args_kfold,
        train_dataset=train_fold_dataset,
        eval_dataset=val_fold_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    trainer.train()
    fold_eval_results = trainer.evaluate()
    all_fold_metrics.append(fold_eval_results)
    print(f"Fold {fold + 1} results: {fold_eval_results}")

print("\n--- K-Fold Cross-Validation Complete ---")


# ============================================================
# 12. K-FOLD RESULTS SUMMARY
# ============================================================
print("\nMetrics summary for each fold:")
for i, metrics in enumerate(all_fold_metrics):
    print(f"Fold {i + 1}: Accuracy={metrics['eval_accuracy']:.4f}, F1-Weighted={metrics['eval_f1_weighted']:.4f}")

avg_accuracy = np.mean([m['eval_accuracy'] for m in all_fold_metrics])
std_accuracy = np.std([m['eval_accuracy'] for m in all_fold_metrics])
avg_f1_weighted = np.mean([m['eval_f1_weighted'] for m in all_fold_metrics])
std_f1_weighted = np.std([m['eval_f1_weighted'] for m in all_fold_metrics])

print(f"\nMean (K={n_splits} Folds) - Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"\nMean (K={n_splits} Folds) - F1-Weighted: {avg_f1_weighted:.4f} ± {std_f1_weighted:.4f}")

# Save K-Fold metrics to JSON
kfold_summary = {
    "n_splits": n_splits,
    "fold_metrics": all_fold_metrics,
    "average": {
        "accuracy": float(avg_accuracy),
        "accuracy_std": float(std_accuracy),
        "f1_weighted": float(avg_f1_weighted),
        "f1_weighted_std": float(std_f1_weighted)
    }
}
kfold_metrics_path = os.path.join(OUTPUT_DIR, "all_fold_metrics.json")
with open(kfold_metrics_path, 'w') as f:
    json.dump(kfold_summary, f, indent=4)
print(f"✓ K-Fold metrics saved to: {kfold_metrics_path}")


# ============================================================
# 13. FINAL MODEL TRAINING ON ORIGINAL TRAIN SET WITH VALIDATION
# ============================================================
print("\n--- Training Final Model (train split + validation for early stopping) ---")

# Use the original train/validation split (same as Baseline) for fair comparison:
# - Train on the train split
# - Use validation split for early stopping & best checkpoint selection
tokenized_train_dataset = dataset[DATASET_CONFIG["split_names"]["train"]].map(tokenize_function, batched=True)
tokenized_val_dataset = dataset[DATASET_CONFIG["split_names"]["validation"]].map(tokenize_function, batched=True)

# Clean columns
for ds_name, ds in [("train", tokenized_train_dataset), ("val", tokenized_val_dataset)]:
    cols_remove = [c for c in ds.column_names if c not in ['input_ids', 'attention_mask', 'labels']]
    if ds_name == "train":
        tokenized_train_dataset = ds.remove_columns(cols_remove)
        tokenized_train_dataset.set_format("torch")
    else:
        tokenized_val_dataset = ds.remove_columns(cols_remove)
        tokenized_val_dataset.set_format("torch")

training_args_final_model = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "final_model_results"),
    learning_rate=best_hyperparameters.get('learning_rate', 2e-5),
    per_device_train_batch_size=best_hyperparameters.get('per_device_train_batch_size', 16),
    per_device_eval_batch_size=best_hyperparameters.get('per_device_train_batch_size', 16),
    num_train_epochs=best_hyperparameters.get('num_train_epochs', 3),
    weight_decay=weight_decay,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    save_total_limit=2,
    logging_dir=os.path.join(OUTPUT_DIR, "final_model_logs"),
    logging_steps=100,
    report_to="tensorboard",
    seed=seed,
)

final_model_trainer = Trainer(
    model_init=model_init,
    args=training_args_final_model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn,
)

final_model_trainer.train()
print("✓ Final training complete (best checkpoint selected via validation)")


# ============================================================
# 14. FINAL EVALUATION ON TEST SET
# ============================================================
print("\n--- Final Evaluation on Test Set ---")

predictions_output = final_model_trainer.predict(tokenized_test_dataset)
logits = predictions_output.predictions
true_labels = predictions_output.label_ids

if is_multilabel:
    predicted_labels = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
else:
    predicted_labels = np.argmax(logits, axis=-1)

print("\nTest Set Metrics:")
final_metrics = compute_metrics_fn((logits, true_labels))
for key, value in final_metrics.items():
    print(f"  {key}: {value:.4f}")

# Save test metrics to JSON
test_metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
with open(test_metrics_path, 'w') as f:
    json.dump(final_metrics, f, indent=4)
print(f"✓ Test metrics saved to: {test_metrics_path}")


# ============================================================
# 15. EMBEDDING VISUALIZATION (centralised)
# ============================================================
visualize_embeddings(
    final_model_trainer.model, tokenized_test_dataset,
    tokenizer, label_names, OUTPUT_DIR,
)


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================
final_model_save_path = os.path.join(OUTPUT_DIR, "final_distilbert_model_kfold")
os.makedirs(final_model_save_path, exist_ok=True)
final_model_trainer.save_model(final_model_save_path)
tokenizer.save_pretrained(final_model_save_path)
print(f"\n✓ Final model and tokenizer saved to: {final_model_save_path}")

print("\n" + "="*60)
print("ALL COMPLETE!")
print("="*60)
print(f"\nResults saved in: {OUTPUT_DIR}/")