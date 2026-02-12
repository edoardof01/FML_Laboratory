# fine_tune_clean.py
"""
Fine-tuning on the Cleaned Dataset (Confident Learning)
========================================================
Same training pipeline as fine_tune_emotions.py but uses the
cleaned dataset produced by clean_dataset.py.
"""
import json
import os
import copy
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    IntervalStrategy
)
from src.data_utils import load_and_preprocess_dataset, tokenize_dataset
from src.metrics import compute_metrics
from src.train_utils import MultiLabelDataCollator, set_seed
from src.viz_utils import visualize_embeddings, plot_label_distribution, plot_confusion_matrix
from src.config_utils import get_config

# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================
config = get_config()
DATASET_CONFIG = copy.deepcopy(config["dataset"])
TRAIN_CONFIG = config["training"]

# Override dataset source to point at the cleaned version
DATASET_CONFIG["path"] = "data/go_emotions_cleaned"
# Remove remote-only keys so load_from_disk is used
DATASET_CONFIG.pop("subset", None)

DATASET_NAME = DATASET_CONFIG["name"].replace("_", "-")
OUTPUT_DIR = f"./outputs/clean_baseline_{DATASET_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seed for reproducibility
set_seed(TRAIN_CONFIG.get("seed", 42))

print(f"\n{'='*60}")
print(f"CLEAN-DATASET BASELINE")
print(f"Dataset (cleaned): {DATASET_CONFIG['path']}")
print(f"OUTPUT: {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ============================================================
# 2. LOAD & PREPROCESS
# ============================================================
dataset, label_info = load_and_preprocess_dataset(DATASET_CONFIG)

label_names = label_info["label_names"]
num_labels = label_info["num_labels"]
is_multilabel = label_info["is_multilabel"]

# ============================================================
# 3. TOKENIZATION
# ============================================================
tokenized_datasets, tokenizer = tokenize_dataset(
    dataset,
    text_column=DATASET_CONFIG["text_column"],
    model_name=config["model"]["name"],
    max_length=config["model"].get("max_length", 128)
)

# ============================================================
# 4. VISUALIZATION (PRE-TRAINING)
# ============================================================
if is_multilabel:
    from collections import Counter
    label_counts = Counter()
    for sample_labels in dataset["train"]["labels"]:
        for idx, val in enumerate(sample_labels):
            if val == 1:
                label_name = label_names[idx]
                label_counts[label_name] += 1
else:
    from collections import Counter
    label_counts = Counter([label_names[l] for l in dataset["train"]["labels"]])

plot_label_distribution(label_counts, OUTPUT_DIR)

from src.model_utils import get_model

# ============================================================
# 5. MODEL INIT
# ============================================================
def model_init(trial=None):
    return get_model(
        model_name=config["model"]["name"],
        num_labels=num_labels,
        is_multilabel=is_multilabel,
        label_names=label_names
    )

# ============================================================
# 6. TRAINING SETUP
# ============================================================
# Data collator
if is_multilabel:
    data_collator = MultiLabelDataCollator(tokenizer=tokenizer, max_length=config["model"].get("max_length", 128))
else:
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics wrapper
def compute_metrics_wrapper(eval_pred):
    return compute_metrics(eval_pred, is_multilabel=is_multilabel)

training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "results"),
    learning_rate=float(TRAIN_CONFIG["learning_rate"]),
    per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=TRAIN_CONFIG["per_device_eval_batch_size"],
    num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
    weight_decay=TRAIN_CONFIG["weight_decay"],
    eval_strategy=TRAIN_CONFIG.get("evaluation_strategy", "epoch"),
    save_strategy=TRAIN_CONFIG.get("save_strategy", "epoch"),
    save_total_limit=TRAIN_CONFIG.get("save_total_limit", 2),
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=TRAIN_CONFIG.get("logging_steps", 100),
    report_to="tensorboard",
    seed=TRAIN_CONFIG.get("seed", 42)
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_wrapper,
)

# ============================================================
# 7. HYPERPARAMETER SEARCH OR TRAINING
# ============================================================
if config.get("optuna", {}).get("enabled", False):
    print("\n--- Starting Hyperparameter Search with Optuna ---")
    best_trial = trainer.hyperparameter_search(
        direction=config["optuna"].get("direction", "maximize"),
        backend="optuna",
        hp_space=lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
        },
        n_trials=config["optuna"].get("n_trials", 3)
    )
    print(f"\n✓ Best trial: {best_trial}")

    # Save best parameters
    with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.json"), 'w') as f:
        json.dump(best_trial.hyperparameters, f, indent=4)

    final_params = best_trial.hyperparameters
else:  # fallback
    final_params = {
        "learning_rate": TRAIN_CONFIG["learning_rate"],
        "per_device_train_batch_size": TRAIN_CONFIG["per_device_train_batch_size"],
        "num_train_epochs": TRAIN_CONFIG["num_train_epochs"]
    }

# ============================================================
# 8. FINAL TRAINING
# ============================================================
print("\n--- Final Training ---")
final_training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "final_results"),
    learning_rate=final_params.get("learning_rate", TRAIN_CONFIG["learning_rate"]),
    per_device_train_batch_size=final_params.get("per_device_train_batch_size", TRAIN_CONFIG["per_device_train_batch_size"]),
    per_device_eval_batch_size=final_params.get("per_device_train_batch_size", TRAIN_CONFIG["per_device_eval_batch_size"]),
    num_train_epochs=final_params.get("num_train_epochs", TRAIN_CONFIG["num_train_epochs"]),
    weight_decay=TRAIN_CONFIG["weight_decay"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_dir=os.path.join(OUTPUT_DIR, "final_logs"),
    logging_steps=100,
    report_to="tensorboard",
    seed=TRAIN_CONFIG.get("seed", 42)
)

final_trainer = Trainer(
    model_init=model_init,
    args=final_training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_wrapper,
)

final_trainer.train()

# ============================================================
# 9. EVALUATION
# ============================================================
print("\n--- Final Evaluation on Test Set ---")
predictions_output = final_trainer.predict(tokenized_datasets["test"])
print("\nTest Set Metrics:", predictions_output.metrics)

if not is_multilabel:
    import numpy as np
    preds = np.argmax(predictions_output.predictions, axis=-1)
    plot_confusion_matrix(predictions_output.label_ids, preds, label_names, OUTPUT_DIR)

visualize_embeddings(final_trainer.model, tokenized_datasets["test"], tokenizer, label_names, OUTPUT_DIR)

final_model_path = os.path.join(OUTPUT_DIR, "final_model")
os.makedirs(final_model_path, exist_ok=True)

# Save test metrics for comparison
with open(os.path.join(final_model_path, "test_metrics.json"), 'w') as f:
    json.dump(predictions_output.metrics, f, indent=4)
print(f"✓ Test metrics saved to {final_model_path}/test_metrics.json")

final_trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\n✓ Model saved to {final_model_path}")
