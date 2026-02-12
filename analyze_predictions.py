#!/usr/bin/env python3
"""
Analyze model predictions to understand prediction patterns.
Specifically checks if models are predicting mostly zeros.
"""
#analyze_predictions.py
import os
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configuration
DATASET_CONFIG = {
    "name": "go_emotions",
    "subset": "simplified",
    "text_column": "text",
    "label_column": "labels",
}

MODEL_PATH = "./kfold_model_go-emotions/final_distilbert_model_kfold"
OUTPUT_DIR = "./prediction_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"{'='*60}")
print("PREDICTION ANALYSIS - K-FOLD MODEL")
print(f"{'='*60}\n")

# Load dataset
print("Loading test dataset...")
dataset = load_dataset(DATASET_CONFIG["name"], DATASET_CONFIG["subset"])
test_dataset = dataset["test"]
train_dataset = dataset["train"]

# Get label info
label_names = train_dataset.features[DATASET_CONFIG["label_column"]].feature.names
num_labels = len(label_names)
print(f"Number of labels: {num_labels}")
print(f"Label names: {label_names}\n")

# Tokenize
print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples[DATASET_CONFIG["text_column"]], 
                    truncation=True, 
                    max_length=128,
                    padding="max_length")

tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask"])

# Load model
print(f"Loading model from {MODEL_PATH}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
model.to(device)

# Get predictions
print("\nGenerating predictions...")
all_logits = []
all_probs = []
all_predictions = []

batch_size = 32
for i in range(0, len(tokenized_test), batch_size):
    batch = tokenized_test[i:i+batch_size]
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_predictions.append(preds.cpu().numpy())

logits_array = np.vstack(all_logits)
probs_array = np.vstack(all_probs)
predictions_array = np.vstack(all_predictions)

# Get true labels - convert from list of indices to binary matrix
print("Converting true labels to binary format...")
raw_labels = test_dataset[DATASET_CONFIG["label_column"]]
true_labels = np.zeros((len(raw_labels), num_labels), dtype=int)
for i, label_indices in enumerate(raw_labels):
    for idx in label_indices:
        if 0 <= idx < num_labels:
            true_labels[i, idx] = 1

print(f"\nPredictions shape: {predictions_array.shape}")
print(f"True labels shape: {true_labels.shape}")

# ANALYSIS 1: Prediction Distribution
print(f"\n{'='*60}")
print("ANALYSIS: Prediction Distribution")
print(f"{'='*60}\n")

total_predictions = predictions_array.sum()
total_possible = predictions_array.size
percentage_positive = (total_predictions / total_possible) * 100

print(f"Total predictions: {predictions_array.shape[0]} samples × {predictions_array.shape[1]} labels = {total_possible}")
print(f"Positive predictions (1s): {total_predictions}")
print(f"Negative predictions (0s): {total_possible - total_predictions}")
print(f"Percentage of 1s: {percentage_positive:.2f}%")

# Per-sample prediction count
predictions_per_sample = predictions_array.sum(axis=1)
print(f"\nPredictions per sample:")
print(f"  Min: {predictions_per_sample.min()}")
print(f"  Max: {predictions_per_sample.max()}")
print(f"  Mean: {predictions_per_sample.mean():.2f}")
print(f"  Median: {np.median(predictions_per_sample):.2f}")

counter = Counter(predictions_per_sample)
print(f"\nDistribution of predictions per sample:")
for count in sorted(counter.keys()):
    print(f"  {count} emotions: {counter[count]} samples ({counter[count]/len(predictions_per_sample)*100:.1f}%)")

# Compare with true labels
true_labels_per_sample = true_labels.sum(axis=1)
print(f"\nTrue labels per sample:")
print(f"  Min: {true_labels_per_sample.min()}")
print(f"  Max: {true_labels_per_sample.max()}")
print(f"  Mean: {true_labels_per_sample.mean():.2f}")
print(f"  Median: {np.median(true_labels_per_sample):.2f}")

# ANALYSIS 2: Per-Label Prediction Frequency
print(f"\n{'='*60}")
print("ANALYSIS: Per-Label Prediction Frequency")
print(f"{'='*60}\n")

per_label_stats = []
for i, label_name in enumerate(label_names):
    pred_count = int(predictions_array[:, i].sum())
    true_count = int(true_labels[:, i].sum())
    mean_prob = float(probs_array[:, i].mean())
    
    per_label_stats.append({
        "label": label_name,
        "predicted_count": pred_count,
        "true_count": true_count,
        "prediction_rate": float(pred_count / len(predictions_array) * 100),
        "true_rate": float(true_count / len(true_labels) * 100),
        "mean_prob": mean_prob
    })

# Sort by prediction rate
per_label_stats.sort(key=lambda x: x["prediction_rate"], reverse=True)

print(f"{'Label':<20} {'Predicted':<12} {'True':<12} {'Pred%':<10} {'True%':<10} {'Mean Prob':<10}")
print("-" * 80)
for stat in per_label_stats:
    print(f"{stat['label']:<20} {stat['predicted_count']:<12} {stat['true_count']:<12} "
          f"{stat['prediction_rate']:<10.2f} {stat['true_rate']:<10.2f} {stat['mean_prob']:<10.4f}")

# ANALYSIS 3: Probability Distribution
print(f"\n{'='*60}")
print("ANALYSIS: Probability Distribution")
print(f"{'='*60}\n")

print(f"Overall probability statistics:")
print(f"  Mean: {probs_array.mean():.4f}")
print(f"  Median: {np.median(probs_array):.4f}")
print(f"  Std: {probs_array.std():.4f}")

# Check concentration near 0.5 threshold
near_threshold = np.abs(probs_array - 0.5) < 0.1
print(f"\nProbabilities near threshold (0.4-0.6): {near_threshold.sum() / probs_array.size * 100:.2f}%")
very_confident_high = (probs_array > 0.9).sum()
very_confident_low = (probs_array < 0.1).sum()
print(f"Very confident predictions:")
print(f"  >0.9: {very_confident_high / probs_array.size * 100:.2f}%")
print(f"  <0.1: {very_confident_low / probs_array.size * 100:.2f}%")

# ANALYSIS 4: Check for class imbalance bias
print(f"\n{'='*60}")
print("ANALYSIS: Class Imbalance Bias Check")
print(f"{'='*60}\n")

# Find rare classes (< 1% in training data)
rare_threshold = 0.01
train_raw_labels = train_dataset[DATASET_CONFIG["label_column"]]
train_labels = np.zeros((len(train_raw_labels), num_labels), dtype=int)
for i, label_indices in enumerate(train_raw_labels):
    for idx in label_indices:
        if 0 <= idx < num_labels:
            train_labels[i, idx] = 1

train_frequencies = train_labels.sum(axis=0) / len(train_labels)

print("Rare classes (< 1% in training):")
for i, (label_name, freq) in enumerate(zip(label_names, train_frequencies)):
    if freq < rare_threshold:
        test_true_count = true_labels[:, i].sum()
        test_pred_count = predictions_array[:, i].sum()
        print(f"  {label_name:<20} Train: {freq*100:.2f}%  Test True: {test_true_count}  Test Pred: {test_pred_count}")

# Save analysis results
results = {
    "total_predictions": int(total_predictions),
    "total_possible": int(total_possible),
    "percentage_positive": float(percentage_positive),
    "predictions_per_sample": {
        "mean": float(predictions_per_sample.mean()),
        "median": float(np.median(predictions_per_sample)),
        "min": int(predictions_per_sample.min()),
        "max": int(predictions_per_sample.max())
    },
    "true_labels_per_sample": {
        "mean": float(true_labels_per_sample.mean()),
        "median": float(np.median(true_labels_per_sample))
    },
    "per_label_stats": per_label_stats,
    "probability_stats": {
        "mean": float(probs_array.mean()),
        "median": float(np.median(probs_array)),
        "std": float(probs_array.std()),
        "near_threshold_pct": float(near_threshold.sum() / probs_array.size * 100),
        "very_confident_high_pct": float(very_confident_high / probs_array.size * 100),
        "very_confident_low_pct": float(very_confident_low / probs_array.size * 100)
    }
}

with open(os.path.join(OUTPUT_DIR, "prediction_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)

# VISUALIZATIONS
print(f"\n{'='*60}")
print("Creating visualizations...")
print(f"{'='*60}\n")

# Plot 1: Predictions per sample distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(predictions_per_sample, bins=range(0, predictions_per_sample.max()+2), 
         edgecolor='black', alpha=0.7)
ax1.axvline(predictions_per_sample.mean(), color='red', linestyle='--', 
            label=f'Mean: {predictions_per_sample.mean():.2f}')
ax1.axvline(true_labels_per_sample.mean(), color='green', linestyle='--', 
            label=f'True Mean: {true_labels_per_sample.mean():.2f}')
ax1.set_xlabel('Number of Emotions Predicted')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Predictions per Sample')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: True vs predicted label frequency
labels_sorted = [s["label"] for s in per_label_stats]
true_counts = [s["true_count"] for s in per_label_stats]
pred_counts = [s["predicted_count"] for s in per_label_stats]

x = np.arange(len(labels_sorted))
width = 0.35

ax2.bar(x - width/2, true_counts, width, label='True', alpha=0.7)
ax2.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
ax2.set_ylabel('Count')
ax2.set_title('True vs Predicted Label Frequency')
ax2.set_xticks(x)
ax2.set_xticklabels(labels_sorted, rotation=90, ha='right')
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_distribution.png"), dpi=150)
print("✓ Saved prediction_distribution.png")

# Plot 3: Probability heatmap (sample of 100 examples)
fig, ax = plt.subplots(figsize=(14, 10))
sample_size = min(100, probs_array.shape[0])
sample_probs = probs_array[:sample_size, :]
sns.heatmap(sample_probs.T, cmap='RdYlGn', vmin=0, vmax=1, 
            yticklabels=label_names, xticklabels=False,
            cbar_kws={'label': 'Probability'})
ax.set_ylabel('Emotions')
ax.set_xlabel(f'Test Samples (first {sample_size})')
ax.set_title('Probability Heatmap (sigmoid outputs)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "probability_heatmap.png"), dpi=150)
print("✓ Saved probability_heatmap.png")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Results saved to: {OUTPUT_DIR}/")
