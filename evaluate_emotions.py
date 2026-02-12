# evaluate_emotions.py
import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

print("=" * 60)
print("Model Comparison - go_emotions (FIXED)")
print("=" * 60)

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATHS = {
    "K-Fold": "./kfold_model_go-emotions/final_distilbert_model_kfold",
    "Optuna": "./optuna_model_go-emotions/final_model", 
    "MLM Domain-Adapted": "./continued_pretraining_go-emotions/final_distilbert_model_with_mlm" 
}

OUTPUT_DIR = "./model_comparison_go-emotions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD DATASET
# ============================================================
print("\n--- Loading Dataset ---")
dataset = load_dataset("go_emotions", "simplified")
test_data = dataset["test"]

# Get label information - CORRECT METHOD
label_names = dataset["train"].features["labels"].feature.names
num_labels = len(label_names)

print(f"✓ Multi-label classification: {num_labels} classes")
print(f"✓ Test set size: {len(test_data)} samples")
print(f"✓ Using device: {DEVICE.type}")

# ============================================================
# CONVERT LABELS TO BINARY VECTORS (MATCHING TRAINING SCRIPTS)
# ============================================================
def convert_to_binary_labels(example):
    """Convert multi-label format to binary vector - MATCHING YOUR TRAINING SCRIPTS"""
    binary_labels = [0] * num_labels
    # Go emotions stores labels as a sequence where 1 indicates presence
    for idx, val in enumerate(example["labels"]):
        if val == 1:
            binary_labels[idx] = 1
    return {"labels_binary": binary_labels}

print("\n--- Converting labels to binary vectors ---")
test_data = test_data.map(convert_to_binary_labels)
labels_true = np.array(test_data["labels_binary"])
print(f"✓ Labels shape: {labels_true.shape}")
print(f"✓ Sample label distribution: {labels_true[0].sum()} labels for first example")

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors=None  # Return lists, not tensors
    )

print("\n--- Tokenizing test set ---")
tokenized_test = test_data.map(tokenize_fn, batched=True)
print("✓ Tokenization complete")

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(model_path, model_name):
    """Evaluate a single model on the test set"""
    if not os.path.exists(model_path):
        print(f"\n⚠ Model '{model_name}' not found at {model_path}. Skipping.")
        return None

    print(f"\n--- Evaluating {model_name} ---")
    try:
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        model.to(DEVICE)
        model.eval()
        print(f"  ✓ Model loaded from {model_path}")

        # Batch prediction
        preds_all = []
        batch_size = 32
        
        with torch.no_grad():
            for i in tqdm(range(0, len(tokenized_test), batch_size), desc=f"  Predicting"):
                batch = tokenized_test[i:i + batch_size]
                inputs = {
                    "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long).to(DEVICE),
                    "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long).to(DEVICE)
                }
                
                outputs = model(**inputs)
                logits = outputs.logits
                # Apply sigmoid for multi-label
                probs = torch.sigmoid(logits).cpu().numpy()
                preds_all.append(probs)

        # Combine predictions
        preds_all = np.concatenate(preds_all, axis=0)
        preds_binary = (preds_all >= 0.5).astype(int)
        
        # Compute metrics
        acc = accuracy_score(labels_true, preds_binary)
        f1_micro = f1_score(labels_true, preds_binary, average="micro", zero_division=0)
        f1_macro = f1_score(labels_true, preds_binary, average="macro", zero_division=0)
        f1_weighted = f1_score(labels_true, preds_binary, average="weighted", zero_division=0)
        precision_weighted = precision_score(labels_true, preds_binary, average="weighted", zero_division=0)
        recall_weighted = recall_score(labels_true, preds_binary, average="weighted", zero_division=0)

        print(f"  ✓ Subset Accuracy: {acc:.4f}")
        print(f"  ✓ F1-micro: {f1_micro:.4f}")
        print(f"  ✓ F1-macro: {f1_macro:.4f}")
        print(f"  ✓ F1-weighted: {f1_weighted:.4f}")
        print(f"  ✓ Precision (weighted): {precision_weighted:.4f}")
        print(f"  ✓ Recall (weighted): {recall_weighted:.4f}")

        return {
            "Model": model_name,
            "Subset_Accuracy": acc,
            "F1_micro": f1_micro,
            "F1_macro": f1_macro,
            "F1_weighted": f1_weighted,
            "Precision_weighted": precision_weighted,
            "Recall_weighted": recall_weighted
        }

    except Exception as e:
        print(f"  ✗ Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# EVALUATE ALL MODELS
# ============================================================
print("\n" + "=" * 60)
print("STARTING MODEL EVALUATION")
print("=" * 60)

results = []

for model_name, model_path in MODEL_PATHS.items():
    metrics = evaluate_model(model_path, model_name)
    if metrics:
        results.append(metrics)

# ============================================================
# SAVE AND DISPLAY RESULTS
# ============================================================
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)

if results:
    df = pd.DataFrame(results)
    
    # Sort by F1-weighted (descending)
    df = df.sort_values("F1_weighted", ascending=False)
    
    # Display with nice formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print("\n" + df.to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✓ Comparison saved to: {csv_path}")
    
    # Highlight best model
    best_model = df.iloc[0]["Model"]
    best_f1 = df.iloc[0]["F1_weighted"]
    print(f"\n🏆 Best Model: {best_model} (F1-weighted: {best_f1:.4f})")
    
else:
    print("\n⚠ No models were successfully evaluated.")
    print("Please check:")
    print("  1. Model paths are correct")
    print("  2. Models were trained and saved properly")
    print("  3. You have the required dependencies installed")

print("\n" + "=" * 80)
print("MODEL COMPARISON COMPLETE")
print("=" * 80)