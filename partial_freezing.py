# partial_freezing.py
import torch
import numpy as np
import optuna
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    IntervalStrategy,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Import src utils to support cleaned dataset loading
from src.config_utils import get_config
from src.data_utils import load_and_preprocess_dataset


# ============================================================
# CONFIGURATION - Load from centralized config  
# ============================================================
config = get_config()
DATASET_CONFIG = config["dataset"]
MODEL_CONFIG = config["model"]

OUTPUT_DIR = f"./layer_analysis_{DATASET_CONFIG['name'].replace('_', '-')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"PARTIAL FREEZING LAYER ANALYSIS")
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


# ============================================================
# 2. TOKENIZATION (USING CENTRALIZED TOKENIZER)
# ============================================================
print("\n--- Tokenizing Dataset ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["name"])

def tokenize_function(examples):
    return tokenizer(examples[DATASET_CONFIG["text_column"]], 
                     truncation=True, 
                     max_length=MODEL_CONFIG.get("max_length", 128))

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove ALL columns except input_ids, attention_mask, and labels
columns_to_remove = [col for col in tokenized_datasets["train"].column_names 
                     if col not in ['input_ids', 'attention_mask', 'labels']]
tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

tokenized_datasets.set_format("torch")
print("✓ Tokenization complete")


# ============================================================
# 5. CUSTOM DATA COLLATOR FOR MULTI-LABEL
# ============================================================
@dataclass
class MultiLabelDataCollator:
    """Data collator for multi-label and single-label classification"""
    tokenizer: Any
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [feature.pop("labels") for feature in features] if "labels" in features[0] else None
        
        batch = self.tokenizer.pad(features, padding=self.padding, return_tensors="pt")
        
        if labels is not None:
            if isinstance(labels[0], torch.Tensor):
                if labels[0].dim() == 0:
                    batch["labels"] = torch.stack(labels)
                else:
                    batch["labels"] = torch.stack(labels).float()
            elif isinstance(labels[0], list):
                batch["labels"] = torch.tensor(labels, dtype=torch.float)
            else:
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return batch


# ============================================================
# 6. METRICS (ADAPTIVE FOR MULTI-LABEL)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    if is_multilabel:
        predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
        labels = labels.astype(int)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        precision = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    else:
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        precision = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "precision_weighted": precision,
        "recall_weighted": recall
    }


# ============================================================
# 7. DATA COLLATOR SETUP
# ============================================================
if is_multilabel:
    data_collator = MultiLabelDataCollator(tokenizer=tokenizer)
    print("✓ Using MultiLabelDataCollator")
else:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("✓ Using standard DataCollatorWithPadding")


# ============================================================
# 8. HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================
if config.get("optuna", {}).get("enabled", False):
    print(f"\n{'='*60}")
    print("--- Hyperparameter Optimization with Optuna ---")
    print(f"{'='*60}\n")
    
    def optuna_objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [16, 32])
        
        # Train with 6 unfrozen layers (full fine-tuning) for HP search
        if is_multilabel:
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_labels,
                problem_type="multi_label_classification"
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_labels
            )
        
        model.config.id2label = {id: label for id, label in enumerate(label_names)}
        model.config.label2id = {label: id for id, label in enumerate(label_names)}
        
        training_args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, f"optuna_trial_{trial.number}"),
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy=IntervalStrategy.EPOCH,
            save_strategy="no",
            logging_steps=100,
            report_to="none",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets[DATASET_CONFIG["split_names"]["train"]],
            eval_dataset=tokenized_datasets[DATASET_CONFIG["split_names"]["validation"]],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        metrics = trainer.evaluate(tokenized_datasets[DATASET_CONFIG["split_names"]["validation"]])
        return metrics["eval_f1_weighted"]
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(optuna_objective, n_trials=config.get("optuna", {}).get("n_trials", 3))
    
    print(f"\n✓ Best hyperparameters: {study.best_trial.params}")
    best_params = study.best_trial.params
    
    with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=4)
else:
    best_params = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 32
    }
    print(f"\n--- Using default hyperparameters (no Optuna) ---")
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Batch size: {best_params['per_device_train_batch_size']}")

# ============================================================
# 9. LAYER ANALYSIS: VARYING UNFROZEN LAYERS
# ============================================================
print(f"\n{'='*60}")
print("--- Layer Analysis: Performance vs Number of Unfrozen Layers ---")
print(f"{'='*60}\n")

results = []
num_transformer_layers = 6  # DistilBERT has 6 layers
num_layers_to_try = list(range(num_transformer_layers + 1))

for num_layers_to_unfreeze in num_layers_to_try:
    print(f"\n***** Training with {num_layers_to_unfreeze} unfrozen layers *****")
    
    # Initialize a new model
    if is_multilabel:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels
        )
    
    model.config.id2label = {id: label for id, label in enumerate(label_names)}
    model.config.label2id = {label: id for id, label in enumerate(label_names)}
    
    # Freeze all parameters initially
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    for name, param in model.named_parameters():
        if 'pre_classifier' in name or 'classifier' in name:
            param.requires_grad = True
    
    # Unfreeze transformer layers from the end
    # DistilBERT layers are numbered 0 to 5
    for layer_idx in range(num_transformer_layers - num_layers_to_unfreeze, num_transformer_layers):
        for name, param in model.named_parameters():
            if f'transformer.layer.{layer_idx}' in name:
                param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Training configuration with optimized hyperparameters
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"results_layers_{num_layers_to_unfreeze}"),
        learning_rate=best_params.get("learning_rate", 2e-5),
        per_device_train_batch_size=best_params.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=best_params.get("per_device_train_batch_size", 32),
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir=os.path.join(OUTPUT_DIR, f"logs_layers_{num_layers_to_unfreeze}"),
        logging_steps=100,
        report_to="tensorboard",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[DATASET_CONFIG["split_names"]["train"]],
        eval_dataset=tokenized_datasets[DATASET_CONFIG["split_names"]["validation"]],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Final evaluation on test set
    predictions_output = trainer.predict(tokenized_datasets[DATASET_CONFIG["split_names"]["test"]])
    test_metrics = predictions_output.metrics
    
    results.append({
        'layers_unfrozen': num_layers_to_unfreeze,
        'f1_weighted': test_metrics['test_f1_weighted'],
        'f1_micro': test_metrics.get('test_f1_micro', test_metrics['test_f1_weighted']),
        'accuracy': test_metrics['test_accuracy'],
        'precision_weighted': test_metrics.get('test_precision_weighted', 0),
        'recall_weighted': test_metrics.get('test_recall_weighted', 0),
        'trainable_params': trainable_params
    })
    
    print(f"Results for {num_layers_to_unfreeze} layers: F1={test_metrics['test_f1_weighted']:.4f}, Accuracy={test_metrics['test_accuracy']:.4f}")


# ============================================================
# 10. VISUALIZATION AND RESULTS
# ============================================================
print(f"\n{'='*60}")
print("--- Results Summary ---")
print(f"{'='*60}\n")

df_results = pd.DataFrame(results)
print(df_results)

# Save results to JSON
results_filepath = os.path.join(OUTPUT_DIR, "layer_analysis_results.json")
with open(results_filepath, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\n✓ Results saved to: {results_filepath}")

# Save best model as final_model for downstream use
best_result = max(results, key=lambda x: x['f1_weighted'])
best_layers = best_result['layers_unfrozen']
print(f"\n--- Saving Best Model (layers={best_layers}, F1={best_result['f1_weighted']:.4f}) ---")

import shutil
best_checkpoint_dir = os.path.join(OUTPUT_DIR, f"results_layers_{best_layers}")
final_model_dir = os.path.join(OUTPUT_DIR, "final_model")

# Find the best checkpoint in the directory
checkpoints = [d for d in os.listdir(best_checkpoint_dir) if d.startswith("checkpoint-")]
if checkpoints:
    # Get the latest checkpoint (highest number)
    best_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    src_path = os.path.join(best_checkpoint_dir, best_checkpoint)
    
    # Copy to final_model
    if os.path.exists(final_model_dir):
        shutil.rmtree(final_model_dir)
    shutil.copytree(src_path, final_model_dir)
    print(f"✓ Best model saved to: {final_model_dir}")

# Plot F1-score vs layers
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_results, x='layers_unfrozen', y='f1_weighted', marker='o', linewidth=2, markersize=8)
plt.title(f'F1-score vs Number of Unfrozen Layers - {DATASET_CONFIG["name"]}')
plt.xlabel('Number of Unfrozen Transformer Layers (Classifier always unfrozen)')
plt.ylabel('F1-score (Weighted)')
plt.xticks(ticks=range(num_transformer_layers + 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_score_vs_layers.png"), dpi=100)
plt.close()
print("✓ F1-score plot saved")

# Plot Accuracy vs layers
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_results, x='layers_unfrozen', y='accuracy', marker='s', linewidth=2, markersize=8, color='green')
plt.title(f'Accuracy vs Number of Unfrozen Layers - {DATASET_CONFIG["name"]}')
plt.xlabel('Number of Unfrozen Transformer Layers (Classifier always unfrozen)')
plt.ylabel('Accuracy')
plt.xticks(ticks=range(num_transformer_layers + 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_layers.png"), dpi=100)
plt.close()
print("✓ Accuracy plot saved")

# Plot trainable parameters
plt.figure(figsize=(12, 7))
sns.barplot(data=df_results, x='layers_unfrozen', y='trainable_params', color='steelblue')
plt.title(f'Trainable Parameters vs Number of Unfrozen Layers - {DATASET_CONFIG["name"]}')
plt.xlabel('Number of Unfrozen Transformer Layers')
plt.ylabel('Trainable Parameters')
plt.xticks(ticks=range(num_transformer_layers + 1))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "trainable_params_vs_layers.png"), dpi=100)
plt.close()
print("✓ Trainable parameters plot saved")

print(f"\n{'='*60}")
print("ALL COMPLETE!")
print(f"{'='*60}")
print(f"\nResults saved in: {OUTPUT_DIR}/")
