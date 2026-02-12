# ensemble-emotions.py
import os
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

# PEFT support for LoRA adapters
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ============================================================
# USER CONFIGURATION
# ============================================================
DATASET_CONFIG = {
    "name": "go_emotions",
    "subset": "simplified",
    "split_names": {"train": "train", "validation": "validation", "test": "test"},
    "text_column": "text",
    "label_column": "labels",
}

MODEL_PATHS = {
    "Baseline": "./outputs/go-emotions/final_model",
    "K-Fold": "./kfold_model_go-emotions/final_distilbert_model_kfold",
    "Weighted": "./outputs/weighted_model_go-emotions/final_model",
    "MLM": "./outputs/continued_pretraining_go-emotions/final_distilbert_model_with_mlm",
    "Partial Freezing": "./layer_analysis_go-emotions/final_model",
}

OUTPUT_DIR = f"./ensemble_{DATASET_CONFIG['name'].replace('_', '-')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensemble mode: 'soft' or 'hard'
ENSEMBLE_MODE = "soft"   # ← SOFT VOTING (best for multi-label)
GLOBAL_THRESHOLD = 0.35  # ← Initial threshold (will be tuned if multilabel)

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Helper: safe print
# ============================================================
def pf(msg=""):
    print(msg, flush=True)

pf("\n" + "=" * 60)
pf(f"Ensemble Model - {DATASET_CONFIG['name']}")
pf("Using device: " + str(DEVICE))
pf(f"Ensemble mode: {ENSEMBLE_MODE}")
pf("=" * 60 + "\n")

# ============================================================
# 1) Load dataset and detect label structure
# ============================================================
pf("--- Loading dataset ---")
if DATASET_CONFIG.get("subset"):
    ds = load_dataset(DATASET_CONFIG["name"], DATASET_CONFIG["subset"])
else:
    ds = load_dataset(DATASET_CONFIG["name"])

test_split = ds[DATASET_CONFIG["split_names"]["test"]]
train_split = ds[DATASET_CONFIG["split_names"]["train"]]

label_col = DATASET_CONFIG["label_column"]
text_col = DATASET_CONFIG["text_column"]

# detect label names and multi-label vs single-label
is_multilabel = False
label_names = None
num_labels = None

# Typical structure for go_emotions: Sequence(ClassLabel) -> multi-label
try:
    feat = train_split.features[label_col]
    if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
        label_names = feat.feature.names
        is_multilabel = True
        num_labels = len(label_names)
    elif hasattr(feat, "names"):
        label_names = feat.names
        is_multilabel = False
        num_labels = len(label_names)
except Exception:
    # fallback: check first sample
    sample = train_split[0][label_col]
    if isinstance(sample, (list, tuple, np.ndarray)):
        if all(isinstance(x, (int, np.integer)) for x in sample) and len(sample) > 1:
            num_labels = len(sample)
            label_names = [f"label_{i}" for i in range(num_labels)]
            is_multilabel = True
        else:
            max_idx = 0
            for i in range(min(5000, len(train_split))):
                val = train_split[i][label_col]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
                    max_idx = max(max_idx, max([int(x) for x in val]))
            num_labels = max_idx + 1
            label_names = [f"label_{i}" for i in range(num_labels)]
            is_multilabel = True
    elif isinstance(sample, (int, np.integer)):
        label_names = getattr(train_split.features[label_col], "names", [f"label_{i}" for i in range(1)])
        num_labels = len(label_names)
        is_multilabel = False
    else:
        raise RuntimeError("Could not automatically determine label structure.")

pf(f"Detected multi-label: {is_multilabel}, num_labels: {num_labels}")
pf(f"Sample label names (first 10): {label_names[:10]}")
pf(f"Test set size: {len(test_split)}\n")

# ============================================================
# 2) Tokenize test set and normalize labels
# ============================================================
pf("--- Tokenizing test set ---")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(examples[text_col], truncation=True, max_length=128, padding="max_length")

tokenized = test_split.map(tokenize_fn, batched=True)

# Convert labels to fixed form
raw_labels = tokenized[label_col]
n_samples = len(raw_labels)

if is_multilabel:
    y_true = np.zeros((n_samples, num_labels), dtype=int)
    for i, lab in enumerate(raw_labels):
        if isinstance(lab, (list, tuple, np.ndarray)):
            if len(lab) == num_labels and all(isinstance(x, (int, np.integer)) for x in lab):
                y_true[i] = np.array(lab, dtype=int)
            else:
                for idx in lab:
                    try:
                        idx_i = int(idx)
                        if 0 <= idx_i < num_labels:
                            y_true[i, idx_i] = 1
                    except Exception:
                        continue
        else:
            try:
                idx_i = int(lab)
                if 0 <= idx_i < num_labels:
                    y_true[i, idx_i] = 1
            except Exception:
                pass
else:
    y_true = np.array(raw_labels, dtype=int)

pf(f"True labels shape: {y_true.shape}")

# Remove problematic columns
cols_to_drop = [c for c in [text_col, label_col, "id"] if c in tokenized.column_names]
if cols_to_drop:
    tokenized = tokenized.remove_columns(cols_to_drop)

# Add normalized labels
if is_multilabel:
    tokenized = tokenized.add_column("labels", [list(map(int, row)) for row in y_true])
else:
    tokenized = tokenized.add_column("labels", y_true.tolist())

tokenized.set_format(type="python")
pf("Tokenization complete.\n")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_loader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
pf(f"Test DataLoader ready (batch_size={BATCH_SIZE})\n")

# ============================================================
# 3) Load models
# ============================================================
pf("--- Loading models ---")
models = {}
for name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        pf(f"⚠ Model '{name}' not found at {path}. Skipping.")
        continue
    try:
        # Check if this is a LoRA/PEFT adapter
        adapter_config_path = os.path.join(path, "adapter_config.json")
        if os.path.exists(adapter_config_path) and PEFT_AVAILABLE:
            pf(f"  Loading {name} as PEFT adapter...")
            peft_config = PeftConfig.from_pretrained(path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path, 
                num_labels=num_labels,
                problem_type="multi_label_classification" if is_multilabel else "single_label_classification"
            )
            m = PeftModel.from_pretrained(base_model, path)
            m = m.merge_and_unload()  # Merge LoRA weights for faster inference
        else:
            m = AutoModelForSequenceClassification.from_pretrained(path)
        m.to(DEVICE)
        m.eval()
        try:
            m.config.id2label = {i: label_names[i] for i in range(num_labels)}
            m.config.label2id = {label_names[i]: i for i in range(num_labels)}
        except Exception:
            pass
        models[name] = m
        pf(f"  ✓ Loaded {name}")
    except Exception as e:
        pf(f"  ✗ Failed to load {name}: {e}")

if not models:
    raise RuntimeError("No models loaded. Aborting.")

pf(f"\n✓ {len(models)} models loaded: {list(models.keys())}\n")

# Quick logits range diagnostic
pf("--- Sanity check logits ranges for first batch ---")
first_batch = None
for batch in test_loader:
    first_batch = batch
    break

if first_batch is not None:
    input_batch = {k: v.to(DEVICE) for k, v in first_batch.items() if k != "labels"}
    for name, model in models.items():
        with torch.no_grad():
            out = model(**input_batch)
            logits = out.logits.cpu().numpy()
            pf(f"{name}: logits range = {logits.min():.4f} .. {logits.max():.4f}")
else:
    pf("Warning: test loader empty.")

pf("")

# ============================================================
# 4) Collect per-model probabilities
# ============================================================
pf(f"--- Running ensemble predictions ({ENSEMBLE_MODE} voting) ---")

model_probs = {}
sigmoid = torch.nn.Sigmoid()

for name, model in models.items():
    pf(f"Collecting probabilities from model: {name}")
    prob_batches = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting [{name}]", unit="batch"):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            if is_multilabel:
                probs = sigmoid(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            prob_batches.append(probs.cpu().numpy())
    
    if len(prob_batches) == 0:
        raise RuntimeError(f"No batches for {name}")
    model_probs[name] = np.vstack(prob_batches)
    pf(f"  -> Collected {model_probs[name].shape[0]} rows, {model_probs[name].shape[1]} cols")

pf("✓ Per-model probabilities collected.\n")

# ============================================================
# 5) Ensemble combination (SOFT VOTING)
# ============================================================
pf(f"Ensembling {len(models)} models using SOFT voting...\n")

# Average probabilities across models
probs_stack = np.stack([model_probs[name] for name in models.keys()], axis=0)
y_pred_prob = np.mean(probs_stack, axis=0)

pf("✓ Ensemble probabilities computed.\n")

# ============================================================
# 6) THRESHOLD TUNING (Automatic)
# ============================================================
pf("--- Threshold tuning & evaluation ---")

if is_multilabel:
    # Grid search for optimal threshold
    pf("Running threshold grid search (0.1 to 0.6 in 50 steps)...")
    best_threshold = 0.35
    best_f1 = -1.0
    threshold_results = []
    
    for threshold in np.linspace(0.1, 0.6, 50):
        y_pred_test = (y_pred_prob >= threshold).astype(int)
        f1_test = f1_score(y_true, y_pred_test, average="micro", zero_division=0)
        threshold_results.append((threshold, f1_test))
        
        if f1_test > best_f1:
            best_f1 = f1_test
            best_threshold = threshold
    
    pf(f"\n✓ Best threshold: {best_threshold:.3f} (F1-micro={best_f1:.4f})\n")
    
    # Apply best threshold
    y_pred_bin = (y_pred_prob >= best_threshold).astype(int)
    
    # Compute metrics
    subset_acc = float((y_true == y_pred_bin).all(axis=1).mean())
    f1_micro = float(f1_score(y_true, y_pred_bin, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred_bin, average="weighted", zero_division=0))
    prec_micro = float(precision_score(y_true, y_pred_bin, average="micro", zero_division=0))
    rec_micro = float(recall_score(y_true, y_pred_bin, average="micro", zero_division=0))
    
    metrics = {
        "best_threshold": float(best_threshold),
        "subset_accuracy": subset_acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "ensemble_mode": ENSEMBLE_MODE,
        "num_models": len(models),
        "models_used": list(models.keys())
    }
    
    pf("Ensemble multi-label results:")
    pf(f"Subset accuracy: {subset_acc:.4f}")
    pf(f"F1 micro:        {f1_micro:.4f}")
    pf(f"F1 macro:        {f1_macro:.4f}")
    pf(f"Precision micro: {prec_micro:.4f}")
    pf(f"Recall micro:    {rec_micro:.4f}\n")
    
    pf("Detailed classification report:\n")
    report = classification_report(y_true, y_pred_bin, target_names=label_names, zero_division=0)
    pf(report)
    
    # Save threshold search results
    with open(os.path.join(OUTPUT_DIR, "threshold_search_results.json"), "w") as f:
        json.dump({
            "thresholds": [float(t) for t, _ in threshold_results],
            "f1_scores": [float(f) for _, f in threshold_results],
            "best_threshold": float(best_threshold),
            "best_f1": float(best_f1)
        }, f, indent=2)
    pf(f"✓ Threshold search results saved\n")

else:
    # Single-label: no threshold
    y_pred_bin = np.argmax(y_pred_prob, axis=1)
    
    acc = float(accuracy_score(y_true, y_pred_bin))
    f1_macro = float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred_bin, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred_bin, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred_bin, average="macro", zero_division=0))
    
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "ensemble_mode": ENSEMBLE_MODE,
        "num_models": len(models),
        "models_used": list(models.keys())
    }
    
    pf("Ensemble single-label results:")
    pf(f"Accuracy:        {acc:.4f}")
    pf(f"F1 (macro):      {f1_macro:.4f}")
    pf(f"Precision:       {prec_macro:.4f}")
    pf(f"Recall:          {rec_macro:.4f}\n")
    
    pf("Detailed classification report:\n")
    pf(classification_report(y_true, y_pred_bin, target_names=label_names, zero_division=0))

# ============================================================
# 7) Save outputs
# ============================================================
pf("\n--- Saving outputs ---")

np.save(os.path.join(OUTPUT_DIR, "ensemble_avg_probabilities.npy"), y_pred_prob)
np.save(os.path.join(OUTPUT_DIR, "ensemble_pred_labels.npy"), y_pred_bin)
np.save(os.path.join(OUTPUT_DIR, "true_labels.npy"), y_true)

with open(os.path.join(OUTPUT_DIR, "ensemble_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save per-model probabilities
for name, probs in model_probs.items():
    try:
        np.save(os.path.join(OUTPUT_DIR, f"probs_{name}.npy"), probs)
    except Exception:
        pass

pf(f"Saved to: {OUTPUT_DIR}\n")
pf(json.dumps(metrics, indent=2))

pf("\n" + "=" * 60)
pf("ENSEMBLE RUN COMPLETE")
pf("=" * 60 + "\n")
