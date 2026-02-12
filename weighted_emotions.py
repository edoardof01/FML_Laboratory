# weighted_emotions.py
import os
import json
import torch
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    IntervalStrategy
)
from src.data_utils import load_and_preprocess_dataset, tokenize_dataset
from src.metrics import compute_metrics
from src.train_utils import MultiLabelDataCollator, WeightedTrainer, set_seed
from src.viz_utils import visualize_embeddings, plot_label_distribution
from src.config_utils import get_config

# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================
config = get_config()
DATASET_CONFIG = config["dataset"]
OUTPUT_DIR = config["output_dir"].replace("go-emotions", "weighted_model_go-emotions")
TRAIN_CONFIG = config["training"]
MODEL_CONFIG = config["model"]

# Ensure output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(TRAIN_CONFIG.get("seed", 42))

OPTUNA_N_TRIALS = config.get("optuna", {}).get("n_trials", 3)

print(f"\n{'='*60}")
print(f"GO-EMOTIONS MULTI-LABEL WEIGHTED CLASSIFICATION")
print(f"DATASET: {DATASET_CONFIG['name']}")
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
    model_name=MODEL_CONFIG["name"],
    max_length=MODEL_CONFIG.get("max_length", 128)
)
tokenized_train = tokenized_datasets["train"]
tokenized_val = tokenized_datasets["validation"]
tokenized_test = tokenized_datasets["test"]

# ============================================================
# 4. CLASS WEIGHTS
# ============================================================
print("\n--- Computing per-class positive weights (pos_weight) ---")
train_labels = tokenized_train["labels"]

# Convert from HuggingFace Column/list to numpy then tensor
if not isinstance(train_labels, torch.Tensor):
    train_labels = torch.tensor(np.array(train_labels), dtype=torch.float)

# Sum across dim 0 (samples) to get counts per label
counts = train_labels.sum(dim=0)
total_examples = len(train_labels)

pos_weight = []
for c in counts:
    neg = total_examples - c.item()
    if c == 0:
        pw = 1.0
    else:
        pw = float(neg) / float(c)
    pos_weight.append(pw)
pos_weight = torch.tensor(pos_weight, dtype=torch.float)

# CRITICAL FIX: Clip extreme pos_weight values to prevent overprediction
# Without clipping, rare classes can have weights of 800+ which causes
# the model to predict positive for almost everything
MAX_POS_WEIGHT = 10.0
pos_weight_before_clip = pos_weight.clone()
pos_weight = torch.clamp(pos_weight, min=0.1, max=MAX_POS_WEIGHT)

# Log statistics about weight clipping
num_clipped = (pos_weight_before_clip > MAX_POS_WEIGHT).sum().item()
if num_clipped > 0:
    print(f"  ⚠ Clipped {num_clipped} pos_weight values (max was {pos_weight_before_clip.max():.1f}, now capped at {MAX_POS_WEIGHT})")

# Save counts
label_counts_dict = {label_names[i]: int(counts[i].item()) for i in range(len(counts))}
with open(os.path.join(OUTPUT_DIR, "label_counts.json"), "w") as f:
    json.dump({"label_names": label_names, "counts": label_counts_dict, "pos_weight": pos_weight.tolist()}, f, indent=2)

print("Label counts saved and pos_weight computed.")
print(f"  Sample pos_weights: {pos_weight[:5].tolist()}")
plot_label_distribution(label_counts_dict, OUTPUT_DIR)


from src.model_utils import get_model

# ============================================================
# 5. MODEL INIT
# ============================================================
def model_init(trial=None):
    return get_model(
        model_name=MODEL_CONFIG["name"],
        num_labels=num_labels,
        is_multilabel=is_multilabel,
        label_names=label_names
    )

# wrapper for metrics
def compute_metrics_wrapper(eval_pred):
    return compute_metrics(eval_pred, is_multilabel=is_multilabel)

# ============================================================
# 6. OPTUNA
# ============================================================
def optuna_objective(trial):
    hp = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }
    
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/optuna_trial_{trial.number}",
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=hp["per_device_train_batch_size"],
        num_train_epochs=hp["num_train_epochs"],
        weight_decay=hp["weight_decay"],
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir=f"{OUTPUT_DIR}/logs_trial_{trial.number}",
        logging_steps=50,
        report_to="tensorboard",
        disable_tqdm=True,
        seed=TRAIN_CONFIG.get("seed", 42),
    )

    trainer = WeightedTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=MultiLabelDataCollator(tokenizer=tokenizer, max_length=MODEL_CONFIG.get("max_length", 128)),
        compute_metrics=compute_metrics_wrapper,
        pos_weight=pos_weight,
    )

    trainer.train()
    metrics = trainer.evaluate(tokenized_val)
    f1 = metrics.get("eval_f1_weighted", metrics.get("f1_weighted"))
    return float(f1)

if config.get("optuna", {}).get("enabled", False):
    print("\n--- Starting Optuna Hyperparameter Search ---")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TRAIN_CONFIG.get("seed", 42)))
    study.optimize(optuna_objective, n_trials=OPTUNA_N_TRIALS)

    print(f"\n--- Optimization Complete ---")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Hyperparameters: {study.best_trial.params}")

    with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.json"), "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
        
    best_hp = study.best_trial.params
else:
    best_hp = {
        "learning_rate": TRAIN_CONFIG["learning_rate"],
        "per_device_train_batch_size": TRAIN_CONFIG["per_device_train_batch_size"],
        "num_train_epochs": TRAIN_CONFIG["num_train_epochs"],
        "weight_decay": TRAIN_CONFIG["weight_decay"]
    }

# ============================================================
# 7. FINAL TRAINING
# ============================================================
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "final_results"),
    learning_rate=best_hp.get("learning_rate", TRAIN_CONFIG["learning_rate"]),
    per_device_train_batch_size=best_hp.get("per_device_train_batch_size", TRAIN_CONFIG["per_device_train_batch_size"]),
    per_device_eval_batch_size=best_hp.get("per_device_train_batch_size", TRAIN_CONFIG["per_device_eval_batch_size"]),
    num_train_epochs=best_hp.get("num_train_epochs", TRAIN_CONFIG["num_train_epochs"]),
    weight_decay=best_hp.get("weight_decay", TRAIN_CONFIG["weight_decay"]),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_dir=os.path.join(OUTPUT_DIR, "final_logs"),
    logging_steps=50,
    report_to="tensorboard",
    seed=TRAIN_CONFIG.get("seed", 42),
)

trainer = WeightedTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=MultiLabelDataCollator(tokenizer=tokenizer, max_length=MODEL_CONFIG.get("max_length", 128)),
    compute_metrics=compute_metrics_wrapper,
    pos_weight=pos_weight,
)

trainer.train()
print("✓ Final training complete")

# ============================================================
# 8. EVALUATION
# ============================================================
print("\n--- Final Evaluation on Test Set ---")
predictions_output = trainer.predict(tokenized_test)
print("\nTest Set Metrics:", predictions_output.metrics)

# Save test metrics to JSON for compare_models.py
test_metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
with open(test_metrics_path, 'w') as f:
    json.dump(predictions_output.metrics, f, indent=4)
print(f"✓ Test metrics saved to: {test_metrics_path}")

# Save predictions
np.save(os.path.join(OUTPUT_DIR, "test_predictions.npy"), predictions_output.predictions)
np.save(os.path.join(OUTPUT_DIR, "test_true_labels.npy"), predictions_output.label_ids)

# Visualization
visualize_embeddings(trainer.model, tokenized_test, tokenizer, label_names, OUTPUT_DIR)

# Save
final_model_save_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_model_save_path)
tokenizer.save_pretrained(final_model_save_path)
print(f"\n✓ Model and tokenizer saved to {final_model_save_path}")
print("\n" + "="*60 + "\nALL COMPLETE!\n" + "="*60)