# k-fold-cross-validation.py
import torch
import numpy as np
from datasets import concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    IntervalStrategy,
    DataCollatorWithPadding
)
from sklearn.model_selection import KFold
import json
import os
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import DataLoader

# Import src utils to support cleaned dataset loading
from src.config_utils import get_config
from src.data_utils import load_and_preprocess_dataset, tokenize_dataset


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
# 2. TOKENIZATION (USING CENTRALIZED TOKENIZER)
# ============================================================
print("\n--- Tokenizing Dataset ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["name"])

def tokenize_function(examples):
    return tokenizer(examples[DATASET_CONFIG["text_column"]], 
                     truncation=True, 
                     max_length=MODEL_CONFIG.get("max_length", 128))

tokenized_full_train_validation_dataset = full_train_validation_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove ALL columns except input_ids, attention_mask, and labels
columns_to_remove = [col for col in tokenized_full_train_validation_dataset.column_names 
                     if col not in ['input_ids', 'attention_mask', 'labels']]
tokenized_full_train_validation_dataset = tokenized_full_train_validation_dataset.remove_columns(columns_to_remove)

columns_to_remove = [col for col in tokenized_test_dataset.column_names 
                     if col not in ['input_ids', 'attention_mask', 'labels']]
tokenized_test_dataset = tokenized_test_dataset.remove_columns(columns_to_remove)

tokenized_full_train_validation_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch")
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
                # Multi-label case
                batch["labels"] = torch.tensor(labels, dtype=torch.float)
            else:
                # Single-label case (scalar integers)
                batch["labels"] = torch.tensor(labels, dtype=torch.long)  # ← LONG, not float!
        
        return batch



# ============================================================
# 6. METRICS (ADAPTIVE FOR MULTI-LABEL)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    if is_multilabel:
        # Multi-label: use sigmoid + threshold
        predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
        labels = labels.astype(int)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        precision = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(labels, predictions, average="weighted", zero_division=0)
        f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
        
    else:
        # Single-label: use argmax
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        precision = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(labels, predictions, average="weighted", zero_division=0)
        f1_micro = f1
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "f1_micro": f1_micro,
        "precision_weighted": precision,
        "recall_weighted": recall
    }


# ============================================================
# 7. MODEL INITIALIZATION (ADAPTIVE)
# ============================================================
def model_init(trial=None):
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
    return model


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
# 9. DATA COLLATOR SETUP
# ============================================================
if is_multilabel:
    data_collator = MultiLabelDataCollator(tokenizer=tokenizer)
    print("✓ Using MultiLabelDataCollator")
else:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("✓ Using standard DataCollatorWithPadding")


# ============================================================
# 10. TRAINING ARGUMENTS
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
        compute_metrics=compute_metrics,
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
    compute_metrics=compute_metrics,
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
final_metrics = compute_metrics((logits, true_labels))
for key, value in final_metrics.items():
    print(f"  {key}: {value:.4f}")

# Save test metrics to JSON
test_metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
with open(test_metrics_path, 'w') as f:
    json.dump(final_metrics, f, indent=4)
print(f"✓ Test metrics saved to: {test_metrics_path}")


# Classification report (only for single-label)
if not is_multilabel:
    print("\n--- Classification Report ---")
    report = classification_report(true_labels, predicted_labels, target_names=label_names, digits=4)
    print(report)
    
    # Confusion matrix
    print("\n--- Generating Normalized Confusion Matrix ---")
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_labels)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)
    
    if num_labels <= 15:
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=label_names)
        disp.plot(cmap="Blues", xticks_rotation='vertical', values_format=".2f")
        plt.title(f'Normalized Confusion Matrix - K-Fold Final Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"))
        plt.close()
        print(f"✓ Confusion matrix saved")
else:
    print("\n(Multi-label dataset: confusion matrix not applicable)")


# ============================================================
# 15. EMBEDDING VISUALIZATION
# ============================================================
print("\n--- Embedding Visualization (PCA & t-SNE) ---")

sample_size = 2000
if len(tokenized_test_dataset) > sample_size:
    sample_indices = np.random.choice(len(tokenized_test_dataset), sample_size, replace=False)
    sampled_dataset = tokenized_test_dataset.select(sample_indices)
else:
    sampled_dataset = tokenized_test_dataset

print(f"Using {len(sampled_dataset)} examples for embedding visualization")

# Use appropriate collator
if is_multilabel:
    embedding_collator = MultiLabelDataCollator(tokenizer=tokenizer)
else:
    embedding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

sampled_dataloader = DataLoader(
    sampled_dataset,
    batch_size=32,
    collate_fn=embedding_collator,
    shuffle=False
)

model = final_model_trainer.model
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_embeddings = []
all_labels_list = []

with torch.no_grad():
    for batch in sampled_dataloader:
        batch_labels = batch.pop('labels')
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True
        )
        sequence_output = outputs.hidden_states[-1][:, 0, :]
        all_embeddings.append(sequence_output.cpu().numpy())
        all_labels_list.append(batch_labels.cpu().numpy())

embeddings_array = np.vstack(all_embeddings)
labels_array = np.vstack(all_labels_list)

print(f"Embeddings shape: {embeddings_array.shape}")
print(f"Labels shape: {labels_array.shape}")

# PCA
print("Running PCA...")
pca = PCA(n_components=2, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_array)

# t-SNE
print("Running t-SNE (this may take a few minutes)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, learning_rate='auto')
embeddings_tsne = tsne.fit_transform(embeddings_array)

# Prepare visualization DataFrames
df_pca = pd.DataFrame(embeddings_pca, columns=['Dim1', 'Dim2'])
df_tsne = pd.DataFrame(embeddings_tsne, columns=['Dim1', 'Dim2'])

if is_multilabel:
    # Get primary emotion (first label with value 1)
    primary_emotions = []
    for label_vec in labels_array:
        label_idx = np.where(label_vec == 1)[0]
        if len(label_idx) > 0:
            primary_emotions.append(label_names[label_idx[0]])
        else:
            primary_emotions.append('unknown')
    df_pca['Emotion'] = primary_emotions
    df_tsne['Emotion'] = primary_emotions
else:
    df_pca['Emotion'] = [label_names[int(label)] for label in labels_array]
    df_tsne['Emotion'] = [label_names[int(label)] for label in labels_array]

# Plot PCA
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='Dim1', y='Dim2',
    hue='Emotion',
    palette='viridis',
    data=df_pca,
    legend='brief',
    alpha=0.7,
    s=20
)
plt.title(f'PCA of DistilBERT Embeddings - {DATASET_CONFIG["name"]}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "embeddings_pca_visualization.png"), dpi=100)
plt.close()
print("✓ PCA visualization saved")

# Plot t-SNE
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='Dim1', y='Dim2',
    hue='Emotion',
    palette='viridis',
    data=df_tsne,
    legend='brief',
    alpha=0.7,
    s=20
)
plt.title(f't-SNE of DistilBERT Embeddings - {DATASET_CONFIG["name"]}')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "embeddings_tsne_visualization.png"), dpi=100)
plt.close()
print("✓ t-SNE visualization saved")


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