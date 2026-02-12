# statistical_significance_analysis.py
"""
Statistical Significance Analysis
==================================
Compares all trained models using:
  1. Bootstrap CIs on multiple metrics (F1 weighted, F1 micro, Hamming accuracy)
  2. Per-label McNemar's test (proper for multi-label)
  3. Paired bootstrap test on F1 differences
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    hamming_loss
)
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
DATASET_CONFIG = {
    "name": "go_emotions",
    "subset": "simplified",
    "split_names": {"train": "train", "validation": "validation", "test": "test"},
    "text_column": "text",
    "label_column": "labels",
}

# All model paths — covers every trained variant
MODEL_PATHS = {
    "Baseline": "./outputs/go-emotions/final_model",
    "K-Fold": "./kfold_model_go-emotions/final_distilbert_model_kfold",
    "Weighted": "./outputs/weighted_model_go-emotions/final_model",
    "MLM": "./outputs/continued_pretraining_go-emotions/final_distilbert_model_with_mlm",
    "Partial Freezing": "./layer_analysis_go-emotions/final_model",
    "Contrastive (SupCon)": "./outputs/go-emotions/supcon_pretraining/final_model",
    "Clean Baseline": "./outputs/clean_baseline_go-emotions/final_model",
}

OUTPUT_DIR = f"./statistical_analysis_{DATASET_CONFIG['name'].replace('_', '-')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"Statistical Significance Analysis for {DATASET_CONFIG['name']}")
print(f"{'='*60}\n")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================
# 1. LOAD DATASET (ADAPTIVE)
# ============================================================
print("--- Loading Test Dataset ---")

if "subset" in DATASET_CONFIG and DATASET_CONFIG["subset"] is not None:
    dataset = load_dataset(DATASET_CONFIG["name"], DATASET_CONFIG["subset"])
else:
    dataset = load_dataset(DATASET_CONFIG["name"])

test_dataset_raw = dataset[DATASET_CONFIG["split_names"]["test"]]
train_split = dataset[DATASET_CONFIG["split_names"]["train"]]

label_column = DATASET_CONFIG["label_column"]
text_column = DATASET_CONFIG["text_column"]

# Detect label structure
if hasattr(train_split.features[label_column], "names"):
    label_names = train_split.features[label_column].names
    num_labels = len(label_names)
    is_multilabel = False
    print(f"✓ Single-label: {num_labels} classes")
elif hasattr(train_split.features[label_column].feature, "names"):
    label_names = train_split.features[label_column].feature.names
    num_labels = len(label_names)
    is_multilabel = True
    print(f"✓ Multi-label: {num_labels} classes")
else:
    raise ValueError("Cannot determine label structure from dataset features.")

print(f"Test set size: {len(test_dataset_raw)}\n")


# ============================================================
# 2. TOKENIZE TEST SET
# ============================================================
print("--- Tokenizing Test Set ---")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples[text_column], truncation=True, max_length=128, padding="max_length")

tokenized_test_dataset = test_dataset_raw.map(tokenize_function, batched=True)

if text_column in tokenized_test_dataset.column_names:
    tokenized_test_dataset = tokenized_test_dataset.remove_columns([text_column])

if label_column != "labels" and label_column in tokenized_test_dataset.column_names:
    tokenized_test_dataset = tokenized_test_dataset.rename_column(label_column, "labels")
    label_column = "labels"

# Extract raw labels BEFORE set_format("torch")
raw_labels = tokenized_test_dataset["labels"]

# Convert labels to binary matrix (n_samples, num_labels)
print("Preparing true labels (binary matrix)...")

if is_multilabel:
    n_samples = len(raw_labels)
    fixed_labels = np.zeros((n_samples, num_labels), dtype=int)
    for i, lab in enumerate(raw_labels):
        if isinstance(lab, (list, tuple, np.ndarray)) and len(lab) == num_labels:
            try:
                fixed_labels[i] = np.array(lab, dtype=int)
            except Exception:
                for j, v in enumerate(lab):
                    fixed_labels[i, j] = int(v)
        else:
            if isinstance(lab, (list, tuple, np.ndarray)):
                for idx in lab:
                    try:
                        idx_int = int(idx)
                        if 0 <= idx_int < num_labels:
                            fixed_labels[i, idx_int] = 1
                    except Exception:
                        continue
            else:
                try:
                    idx_int = int(lab)
                    if 0 <= idx_int < num_labels:
                        fixed_labels[i, idx_int] = 1
                except Exception:
                    pass
    true_labels = fixed_labels
else:
    true_labels = np.array(raw_labels, dtype=int)

print(f"True labels shape: {true_labels.shape}\n")

tokenized_test_dataset.set_format("torch")
print("✓ Tokenization complete\n")

# Save true labels & metadata
np.save(os.path.join(OUTPUT_DIR, "true_labels.npy"), true_labels)
with open(os.path.join(OUTPUT_DIR, "meta_label_names.json"), "w") as f:
    json.dump({"label_names": label_names, "num_labels": int(num_labels),
               "is_multilabel": bool(is_multilabel)}, f, indent=2)


# ============================================================
# METRIC HELPERS
# ============================================================
def compute_all_metrics(y_true, y_pred, is_ml):
    """Compute a dict of metrics for multi-label or single-label."""
    metrics = {}
    if is_ml:
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["hamming_accuracy"] = 1.0 - hamming_loss(y_true, y_pred)
        metrics["subset_accuracy"] = accuracy_score(y_true, y_pred)
    else:
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    return metrics


# ============================================================
# 3. GET PREDICTIONS FROM ALL MODELS
# ============================================================
print("--- Generating Predictions from All Models ---\n")
all_predictions = {}

for model_name, model_path in MODEL_PATHS.items():
    if not os.path.exists(model_path):
        print(f"⚠ Model '{model_name}' not found at {model_path}. Skipping.")
        continue

    try:
        print(f"  Loading {model_name} from {model_path} ...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model.to(device)

        batch_size = 32
        preds_batches = []

        for i in range(0, len(tokenized_test_dataset), batch_size):
            batch = tokenized_test_dataset[i : i + batch_size]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            if is_multilabel:
                batch_preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            else:
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()

            preds_batches.append(batch_preds)

        if len(preds_batches) == 0:
            predictions = np.zeros((0, num_labels), dtype=int) if is_multilabel else np.zeros((0,), dtype=int)
        else:
            predictions = np.vstack(preds_batches) if is_multilabel else np.concatenate(preds_batches)

        # Harmonize column count if needed
        if is_multilabel and predictions.ndim == 2 and predictions.shape[1] != num_labels:
            if predictions.shape[1] < num_labels:
                pad_width = num_labels - predictions.shape[1]
                predictions = np.pad(predictions, ((0, 0), (0, pad_width)), constant_values=0)
            else:
                predictions = predictions[:, :num_labels]

        if predictions.shape[0] != len(tokenized_test_dataset):
            print(f"    ⚠ Warning: predictions length ({predictions.shape[0]}) != test size ({len(tokenized_test_dataset)})")

        all_predictions[model_name] = predictions

        # Quick metric summary
        m = compute_all_metrics(true_labels, predictions, is_multilabel)
        if is_multilabel:
            print(f"    ✓ {model_name}  F1w={m['f1_weighted']:.4f}  F1μ={m['f1_micro']:.4f}  "
                  f"Hamming={m['hamming_accuracy']:.4f}  Subset={m['subset_accuracy']:.4f}\n")
        else:
            print(f"    ✓ {model_name}  F1w={m['f1_weighted']:.4f}  Acc={m['accuracy']:.4f}\n")

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"    ✗ Error loading/predicting with '{model_name}': {e}\n")

if len(all_predictions) < 2:
    print("ERROR: Need at least 2 models to perform statistical tests. Exiting.")
    sys.exit(1)

model_names = list(all_predictions.keys())
n_samples = len(true_labels)
print(f"\n✓ Successfully loaded {len(model_names)} models: {model_names}\n")


# ============================================================
# 4. PER-LABEL McNEMAR'S TEST (proper for multi-label)
# ============================================================
print("\n--- Per-Label McNemar's Test ---\n")

if is_multilabel:
    # For each label, run pairwise McNemar between all model pairs
    # Then aggregate: median p-value across labels, and count of
    # labels where the difference is significant (p < 0.05)
    perlabel_results = {}

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i >= j:
                continue
            pair_key = f"{m1} vs {m2}"
            label_pvalues = []

            for k in range(num_labels):
                preds1_k = all_predictions[m1][:, k]
                preds2_k = all_predictions[m2][:, k]
                true_k = true_labels[:, k]

                correct1 = (preds1_k == true_k)
                correct2 = (preds2_k == true_k)

                n00 = int(np.sum((~correct1) & (~correct2)))
                n01 = int(np.sum(correct1 & (~correct2)))
                n10 = int(np.sum((~correct1) & correct2))
                n11 = int(np.sum(correct1 & correct2))

                table = [[n00, n01], [n10, n11]]

                # Need n01 + n10 > 0 for a meaningful test
                if n01 + n10 == 0:
                    label_pvalues.append(1.0)
                    continue

                try:
                    use_exact = (n01 + n10) < 25
                    result = mcnemar(table, exact=use_exact, correction=not use_exact)
                    label_pvalues.append(float(result.pvalue))
                except Exception:
                    label_pvalues.append(1.0)

            label_pvalues = np.array(label_pvalues)
            n_significant = int(np.sum(label_pvalues < 0.05))
            median_p = float(np.median(label_pvalues))
            mean_p = float(np.mean(label_pvalues))

            perlabel_results[pair_key] = {
                "per_label_pvalues": {label_names[k]: float(label_pvalues[k]) for k in range(num_labels)},
                "n_significant_labels": n_significant,
                "n_total_labels": num_labels,
                "pct_significant": round(100.0 * n_significant / num_labels, 1),
                "median_pvalue": median_p,
                "mean_pvalue": mean_p,
            }

            sig_str = f"{n_significant}/{num_labels} labels significant"
            print(f"  {pair_key}: {sig_str} (median p={median_p:.4f})")

    # Save per-label McNemar results
    with open(os.path.join(OUTPUT_DIR, "mcnemar_perlabel_results.json"), "w") as f:
        json.dump(perlabel_results, f, indent=2)
    print(f"\n  Saved per-label McNemar results.\n")
else:
    perlabel_results = None


# ============================================================
# 5. AGGREGATE McNEMAR HEATMAP (sample-level, Hamming-style)
# ============================================================
print("--- Aggregate McNemar's Test (per-sample Hamming correctness) ---\n")

mcnemar_table = pd.DataFrame(index=model_names, columns=model_names)
p_value_table = pd.DataFrame(index=model_names, columns=model_names, dtype=float)

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if i >= j:
            mcnemar_table.loc[m1, m2] = "-"
            p_value_table.loc[m1, m2] = np.nan
            continue

        preds1 = all_predictions[m1]
        preds2 = all_predictions[m2]

        if is_multilabel:
            correct1_flat = (preds1 == true_labels).ravel()
            correct2_flat = (preds2 == true_labels).ravel()
        else:
            correct1_flat = (preds1 == true_labels)
            correct2_flat = (preds2 == true_labels)

        n01 = int(np.sum(correct1_flat & (~correct2_flat)))
        n10 = int(np.sum((~correct1_flat) & correct2_flat))
        n00 = int(np.sum((~correct1_flat) & (~correct2_flat)))
        n11 = int(np.sum(correct1_flat & correct2_flat))

        table = [[n00, n01], [n10, n11]]

        try:
            result = mcnemar(table, exact=False, correction=True)
            p_value = float(result.pvalue)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            mcnemar_table.loc[m1, m2] = f"p={p_value:.4f} {sig}"
            p_value_table.loc[m1, m2] = p_value
            p_value_table.loc[m2, m1] = p_value

            print(f"  {m1} vs {m2}: p={p_value:.6f} {sig}")
        except Exception as e:
            print(f"  ✗ McNemar test failed for {m1} vs {m2}: {e}")
            mcnemar_table.loc[m1, m2] = "error"
            p_value_table.loc[m1, m2] = np.nan
            p_value_table.loc[m2, m1] = np.nan

mcnemar_filepath = os.path.join(OUTPUT_DIR, "mcnemar_test_results.csv")
mcnemar_table.to_csv(mcnemar_filepath)
print(f"\n  Saved to {mcnemar_filepath}\n")


# ============================================================
# 6. BOOTSTRAP CONFIDENCE INTERVALS (multiple metrics)
# ============================================================
print("--- Bootstrap Confidence Intervals (95%) ---\n")

# Metrics to bootstrap
if is_multilabel:
    BOOT_METRICS = ["f1_weighted", "f1_micro", "hamming_accuracy", "subset_accuracy"]
else:
    BOOT_METRICS = ["f1_weighted", "f1_micro", "accuracy"]

bootstrap_results = {}
n_bootstrap = 1000
np.random.seed(42)

for model_name, preds in all_predictions.items():
    print(f"  Bootstrapping {model_name} ...")
    boot_scores = {m: [] for m in BOOT_METRICS}

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_b = true_labels[indices]
        y_pred_b = preds[indices]
        m = compute_all_metrics(y_true_b, y_pred_b, is_multilabel)
        for metric_name in BOOT_METRICS:
            boot_scores[metric_name].append(m[metric_name])

    # Overall (non-bootstrapped) metrics
    overall = compute_all_metrics(true_labels, preds, is_multilabel)

    model_result = {}
    for metric_name in BOOT_METRICS:
        arr = np.array(boot_scores[metric_name])
        model_result[metric_name] = {
            "value": overall[metric_name],
            "mean_bootstrap": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "std": float(np.std(arr)),
        }

    bootstrap_results[model_name] = model_result

    # Print key metrics
    fw = model_result["f1_weighted"]
    fm = model_result["f1_micro"]
    print(f"    F1w={fw['value']:.4f} [{fw['ci_lower']:.4f}, {fw['ci_upper']:.4f}]  "
          f"F1μ={fm['value']:.4f} [{fm['ci_lower']:.4f}, {fm['ci_upper']:.4f}]")
    if is_multilabel:
        ha = model_result["hamming_accuracy"]
        print(f"    Hamming={ha['value']:.4f} [{ha['ci_lower']:.4f}, {ha['ci_upper']:.4f}]")
    print()

with open(os.path.join(OUTPUT_DIR, "bootstrap_results.json"), "w") as f:
    json.dump(bootstrap_results, f, indent=2)


# ============================================================
# 7. PAIRED BOOTSTRAP TEST ON F1 DIFFERENCES
# ============================================================
print("--- Paired Bootstrap Test (F1 Weighted Differences) ---\n")

paired_bootstrap_results = {}
np.random.seed(42)

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if i >= j:
            continue

        preds1 = all_predictions[m1]
        preds2 = all_predictions[m2]

        # Observed F1 difference
        f1_1 = f1_score(true_labels, preds1, average="weighted", zero_division=0)
        f1_2 = f1_score(true_labels, preds2, average="weighted", zero_division=0)
        observed_diff = f1_1 - f1_2

        # Bootstrap the difference
        diffs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_b = true_labels[indices]
            f1_b1 = f1_score(y_true_b, preds1[indices], average="weighted", zero_division=0)
            f1_b2 = f1_score(y_true_b, preds2[indices], average="weighted", zero_division=0)
            diffs.append(f1_b1 - f1_b2)

        diffs = np.array(diffs)
        ci_lower = float(np.percentile(diffs, 2.5))
        ci_upper = float(np.percentile(diffs, 97.5))
        # p-value: proportion of bootstrap samples where difference changes sign
        if observed_diff >= 0:
            p_value = float(np.mean(diffs < 0)) * 2  # two-sided
        else:
            p_value = float(np.mean(diffs > 0)) * 2
        p_value = min(p_value, 1.0)

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)

        pair_key = f"{m1} vs {m2}"
        paired_bootstrap_results[pair_key] = {
            "model_1": m1,
            "model_2": m2,
            "f1_model_1": float(f1_1),
            "f1_model_2": float(f1_2),
            "observed_diff": float(observed_diff),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "ci_excludes_zero": ci_excludes_zero,
        }

        print(f"  {pair_key}: Δ={observed_diff:+.4f} [{ci_lower:+.4f}, {ci_upper:+.4f}] p={p_value:.4f} {sig}")

with open(os.path.join(OUTPUT_DIR, "paired_bootstrap_f1_results.json"), "w") as f:
    json.dump(paired_bootstrap_results, f, indent=2)
print()


# ============================================================
# 8. SUMMARY TABLE
# ============================================================
print("--- Summary Table ---\n")
summary_rows = []
for model_name in model_names:
    r = bootstrap_results.get(model_name, {})
    row = {"Model": model_name}
    for metric_name in BOOT_METRICS:
        mr = r.get(metric_name, {})
        row[metric_name] = f"{mr.get('value', float('nan')):.4f}"
        row[f"{metric_name}_CI"] = (f"[{mr.get('ci_lower', float('nan')):.4f}, "
                                     f"{mr.get('ci_upper', float('nan')):.4f}]")
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_filepath = os.path.join(OUTPUT_DIR, "bootstrap_ci_summary.csv")
summary_df.to_csv(summary_filepath, index=False)
print(f"\nSaved summary to {summary_filepath}\n")


# ============================================================
# 9. VISUALIZATIONS
# ============================================================
print("--- Generating Visualizations ---\n")

# --- Helper: determine subplot grid layout ---
def subplot_grid(n):
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    return rows, cols

# ----- Plot 1: Bootstrap F1 Weighted distributions -----
primary_metric = "f1_weighted"
n_models = len(model_names)
rows, cols = subplot_grid(n_models)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
if n_models == 1:
    axes = [axes]
else:
    axes = np.array(axes).flatten()

np.random.seed(42)
for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    preds = all_predictions[model_name]
    scores = []
    for _ in range(1000):
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        scores.append(f1_score(true_labels[idxs], preds[idxs], average="weighted", zero_division=0))
    ax.hist(scores, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    val = bootstrap_results[model_name][primary_metric]["value"]
    ax.axvline(val, color="red", linestyle="--", linewidth=2, label=f"observed={val:.4f}")
    ax.set_title(model_name, fontsize=11)
    ax.set_xlabel("F1 Weighted")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

for idx in range(n_models, rows * cols):
    fig.delaxes(axes[idx])

plt.suptitle("Bootstrap Distributions — F1 Weighted", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bootstrap_distributions.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ Bootstrap distributions saved")

# ----- Plot 2: Multi-metric comparison with CI -----
fig, axes_plot = plt.subplots(1, len(BOOT_METRICS), figsize=(6 * len(BOOT_METRICS), 7))
if len(BOOT_METRICS) == 1:
    axes_plot = [axes_plot]

colors = plt.cm.tab10(np.linspace(0, 1, n_models))

for ax, metric_name in zip(axes_plot, BOOT_METRICS):
    values = [bootstrap_results[m][metric_name]["value"] for m in model_names]
    ci_l = [bootstrap_results[m][metric_name]["ci_lower"] for m in model_names]
    ci_u = [bootstrap_results[m][metric_name]["ci_upper"] for m in model_names]
    errors_lower = [values[k] - ci_l[k] for k in range(n_models)]
    errors_upper = [ci_u[k] - values[k] for k in range(n_models)]

    bars = ax.bar(range(n_models), values,
                  yerr=[errors_lower, errors_upper],
                  capsize=5, alpha=0.8, edgecolor="black", color=colors)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name.replace("_", " ").title() + " with 95% CI", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for k, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.001,
                f"{values[k]:.4f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison_ci.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ Multi-metric comparison plot saved")

# ----- Plot 3: McNemar heatmap (label-level) -----
if len(model_names) > 1:
    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), max(6, n_models * 1.2)))
    heatmap_data = p_value_table.astype(float)
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="RdYlGn_r",
                cbar_kws={"label": "p-value"},
                ax=ax, vmin=0, vmax=0.1, square=True, linewidths=0.5)
    ax.set_title("McNemar's Test P-values (label-level flattened)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mcnemar_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ McNemar's heatmap saved")

# ----- Plot 4: Per-label McNemar significance summary -----
if is_multilabel and perlabel_results:
    pair_keys = list(perlabel_results.keys())
    n_sig = [perlabel_results[k]["n_significant_labels"] for k in pair_keys]

    fig, ax = plt.subplots(figsize=(max(10, len(pair_keys) * 1.2), 6))
    bar_colors = ["#d32f2f" if s > num_labels / 2 else "#ffa726" if s > 5 else "#66bb6a" for s in n_sig]
    bars = ax.barh(range(len(pair_keys)), n_sig, color=bar_colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(pair_keys)))
    ax.set_yticklabels(pair_keys, fontsize=9)
    ax.set_xlabel(f"Number of significant labels (out of {num_labels})")
    ax.set_title("Per-Label McNemar: Significant Differences (p < 0.05)")
    ax.axvline(num_labels / 2, color="gray", linestyle="--", alpha=0.5, label=f"50% ({num_labels // 2})")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    for k, bar in enumerate(bars):
        w = bar.get_width()
        pct = perlabel_results[pair_keys[k]]["pct_significant"]
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2.,
                f"{int(w)} ({pct}%)", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mcnemar_perlabel_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Per-label McNemar summary plot saved")

# ----- Plot 5: Paired bootstrap F1 difference heatmap -----
if paired_bootstrap_results:
    diff_matrix = pd.DataFrame(np.zeros((n_models, n_models)),
                                index=model_names, columns=model_names)
    sig_matrix = pd.DataFrame("", index=model_names, columns=model_names)

    for pair_key, res in paired_bootstrap_results.items():
        m1, m2 = res["model_1"], res["model_2"]
        diff_matrix.loc[m1, m2] = res["observed_diff"]
        diff_matrix.loc[m2, m1] = -res["observed_diff"]
        sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else ""
        sig_matrix.loc[m1, m2] = f"{res['observed_diff']:+.3f}{sig}"
        sig_matrix.loc[m2, m1] = f"{-res['observed_diff']:+.3f}{sig}"

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), max(6, n_models * 1.2)))
    max_abs = max(abs(diff_matrix.values.min()), abs(diff_matrix.values.max()), 0.01)
    sns.heatmap(diff_matrix.astype(float), annot=sig_matrix.values, fmt="",
                cmap="RdBu_r", center=0, vmin=-max_abs, vmax=max_abs,
                cbar_kws={"label": "F1 Weighted Δ (row − col)"},
                ax=ax, square=True, linewidths=0.5)
    ax.set_title("Paired Bootstrap: F1 Weighted Differences (row − column)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "paired_bootstrap_f1_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Paired bootstrap F1 heatmap saved")


print(f"\n{'='*60}")
print("STATISTICAL ANALYSIS COMPLETE")
print(f"{'='*60}\n")
print(f"Results saved in: {OUTPUT_DIR}/")
print(f"  - bootstrap_results.json           (CIs for F1, Hamming, etc.)")
print(f"  - bootstrap_ci_summary.csv         (summary table)")
print(f"  - mcnemar_test_results.csv         (aggregate label-level McNemar)")
print(f"  - mcnemar_perlabel_results.json    (per-label McNemar)")
print(f"  - paired_bootstrap_f1_results.json (paired F1 difference tests)")
print(f"  - *.png                            (visualizations)")
print()
