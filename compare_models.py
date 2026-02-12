# compare_models.py
"""
Model Comparison Script
=======================
Collects metrics from all trained models and generates:
1. Comparison table (CSV)
2. Comparison charts (PNG)
3. Summary report (JSON)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.config_utils import get_config

# ============================================================
# 1. SETUP
# ============================================================
config = get_config()
DATASET_NAME = config["dataset"]["name"].replace("_", "-")
OUTPUT_DIR = f"./model_comparison_{DATASET_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print("MODEL COMPARISON ANALYSIS")
print(f"{'='*60}\n")

# ============================================================
# 2. DEFINE MODEL DIRECTORIES AND EXPECTED METRIC FILES
# ============================================================
MODEL_CONFIGS = {
    "Baseline (Fine-tune)": {
        "dir": f"./outputs/{DATASET_NAME}",
        "metrics_file": "final_model/test_metrics.json",
        "alt_metrics_file": None
    },
    "K-Fold CV": {
        "dir": f"./kfold_model_{DATASET_NAME}",
        "metrics_file": "test_metrics.json",
        "alt_metrics_file": "all_fold_metrics.json"
    },
    "Weighted Classes": {
        "dir": f"./outputs/weighted_model_{DATASET_NAME}",
        "metrics_file": "test_metrics.json",
        "alt_metrics_file": "final_results/trainer_state.json"
    },
    "Partial Freezing": {
        "dir": f"./layer_analysis_{DATASET_NAME}",
        "metrics_file": "layer_analysis_results.json",
        "alt_metrics_file": None
    },
    "MLM Pretraining": {
        "dir": f"./outputs/continued_pretraining_{DATASET_NAME}",
        "metrics_file": "test_metrics.json",
        "alt_metrics_file": "finetuning_results/trainer_state.json"
    },
    "Contrastive (SupCon)": {
        "dir": f"./outputs/{DATASET_NAME}/supcon_pretraining",
        "metrics_file": "test_metrics.json",
        "alt_metrics_file": None
    },
    "Ensemble": {
        "dir": f"./ensemble_{DATASET_NAME}",
        "metrics_file": "ensemble_metrics.json",
        "alt_metrics_file": None
    },
    "Clean Baseline": {
        "dir": f"./outputs/clean_baseline_{DATASET_NAME}",
        "metrics_file": "final_model/test_metrics.json",
        "alt_metrics_file": None
    }
}

# ============================================================
# 3. COLLECT METRICS FROM ALL MODELS
# ============================================================
def find_metrics_file(base_dir):
    """Search for any JSON file containing metrics in the directory tree."""
    metrics_keywords = ["metrics", "results", "evaluation", "test"]
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.json'):
                if any(kw in f.lower() for kw in metrics_keywords):
                    return os.path.join(root, f)
    
    # Fallback: look for trainer_state.json
    for root, dirs, files in os.walk(base_dir):
        if "trainer_state.json" in files:
            return os.path.join(root, "trainer_state.json")
    
    return None

def extract_metrics_from_trainer_state(filepath):
    """Extract metrics from HuggingFace trainer_state.json"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get the last evaluation metrics
    log_history = data.get("log_history", [])
    metrics = {}
    
    for entry in reversed(log_history):
        if "eval_f1_micro" in entry or "eval_f1_weighted" in entry or "eval_accuracy" in entry:
            for k, v in entry.items():
                if k.startswith("eval_"):
                    new_key = k.replace("eval_", "test_")
                    metrics[new_key] = v
            break
    
    return metrics

def load_model_metrics(model_name, model_config):
    """Load metrics for a single model."""
    base_dir = model_config["dir"]
    
    if not os.path.exists(base_dir):
        print(f"  ⚠ Directory not found: {base_dir}")
        return None
    
    def process_json_data(data):
        """Process JSON data, handling both dict and list formats."""
        if isinstance(data, dict):
            return data
        elif isinstance(data, list) and len(data) > 0:
            # For layer analysis results: pick the best result (highest f1_weighted with layers > 0)
            best_result = None
            best_f1 = -1
            for item in data:
                if isinstance(item, dict):
                    # Skip layers_unfrozen=0 as it's essentially no training
                    if item.get("layers_unfrozen", 1) == 0:
                        continue
                    f1 = item.get("f1_weighted", item.get("f1_micro", 0))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_result = item
            return best_result if best_result else (data[0] if isinstance(data[0], dict) else None)
        return None
    
    # Try primary metrics file
    if model_config["metrics_file"]:
        primary_path = os.path.join(base_dir, model_config["metrics_file"])
        if os.path.exists(primary_path):
            try:
                with open(primary_path, 'r') as f:
                    data = json.load(f)
                    if "trainer_state" in model_config["metrics_file"]:
                        return extract_metrics_from_trainer_state(primary_path)
                    return process_json_data(data)
            except:
                pass
    
    # Try alternative metrics file
    if model_config["alt_metrics_file"]:
        alt_path = os.path.join(base_dir, model_config["alt_metrics_file"])
        if os.path.exists(alt_path):
            try:
                with open(alt_path, 'r') as f:
                    return process_json_data(json.load(f))
            except:
                pass
    
    # Search for any metrics file
    found_file = find_metrics_file(base_dir)
    if found_file:
        try:
            if "trainer_state" in found_file:
                return extract_metrics_from_trainer_state(found_file)
            else:
                with open(found_file, 'r') as f:
                    return process_json_data(json.load(f))
        except:
            pass
    
    return None

print("--- Collecting Metrics from All Models ---\n")
all_metrics = {}

for model_name, model_config in MODEL_CONFIGS.items():
    print(f"  Loading: {model_name}...")
    metrics = load_model_metrics(model_name, model_config)
    
    if metrics:
        all_metrics[model_name] = metrics
        print(f"    ✓ Found metrics")
    else:
        print(f"    ✗ No metrics found")

print(f"\n✓ Collected metrics from {len(all_metrics)} models\n")

if len(all_metrics) == 0:
    print("ERROR: No metrics found from any model!")
    print("Make sure you've trained at least one model before running comparison.")
    exit(1)

# ============================================================
# 4. NORMALIZE AND AGGREGATE METRICS
# ============================================================
def normalize_metric_name(key):
    """Normalize metric names for comparison."""
    key = key.lower().replace("eval_", "").replace("test_", "")
    
    # Standard mappings
    mappings = {
        "f1_weighted": "F1 Weighted",
        "f1_micro": "F1 Micro",
        "f1_macro": "F1 Macro",
        "accuracy": "Accuracy",
        "subset_accuracy": "Accuracy",  # Ensemble uses this name
        "precision_weighted": "Precision",
        "recall_weighted": "Recall",
        "precision_micro": "Precision",  # Fallback for ensemble
        "recall_micro": "Recall",  # Fallback for ensemble
    }
    
    return mappings.get(key, key.title())

# Standard metrics to compare
METRICS_TO_COMPARE = [
    "F1 Weighted", "F1 Micro", "Accuracy", "Precision", "Recall"
]

comparison_data = []

for model_name, metrics in all_metrics.items():
    row = {"Model": model_name}
    
    for orig_key, value in metrics.items():
        norm_key = normalize_metric_name(orig_key)
        if isinstance(value, (int, float)) and not np.isnan(value):
            row[norm_key] = value
    
    comparison_data.append(row)

df = pd.DataFrame(comparison_data)

# Keep only common metrics
available_metrics = [m for m in METRICS_TO_COMPARE if m in df.columns]
df = df[["Model"] + available_metrics].dropna(axis=1, how='all')

# Sort by F1 Weighted (or first available metric)
sort_col = available_metrics[0] if available_metrics else df.columns[1]
df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

print("--- Comparison Table ---\n")
print(df.to_string(index=False))
print()

# ============================================================
# 5. SAVE COMPARISON TABLE
# ============================================================
csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
df.to_csv(csv_path, index=False)
print(f"✓ Comparison table saved to: {csv_path}")

# ============================================================
# 6. GENERATE COMPARISON CHARTS
# ============================================================
print("\n--- Generating Comparison Charts ---")

# Chart 1: Bar chart of all metrics
fig, ax = plt.subplots(figsize=(14, 8))
df_melted = df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", ax=ax)
ax.set_title("Model Performance Comparison", fontsize=16, fontweight='bold')
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc='upper left')
ax.set_ylim(0, 1)
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "model_comparison_bar.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Bar chart saved: {chart_path}")

# Chart 2: Heatmap
if len(df) > 1 and len(available_metrics) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_data = df.set_index("Model")[available_metrics]
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title("Model Performance Heatmap", fontsize=16, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "model_comparison_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap saved: {heatmap_path}")

# Chart 3: Radar chart (if enough metrics)
if len(available_metrics) >= 3 and len(df) <= 8:
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    categories = available_metrics
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for idx, row in df.iterrows():
        values = row[available_metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row["Model"], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Model Performance Radar Chart", fontsize=14, fontweight='bold', y=1.08)
    
    radar_path = os.path.join(OUTPUT_DIR, "model_comparison_radar.png")
    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Radar chart saved: {radar_path}")

# ============================================================
# 7. IDENTIFY BEST MODEL
# ============================================================
print("\n--- Best Model Analysis ---\n")

best_models = {}
for metric in available_metrics:
    if metric in df.columns:
        best_idx = df[metric].idxmax()
        best_model = df.loc[best_idx, "Model"]
        best_score = df.loc[best_idx, metric]
        best_models[metric] = {"model": best_model, "score": best_score}
        print(f"  Best {metric}: {best_model} ({best_score:.4f})")

# Overall best (by average rank)
if len(available_metrics) > 1:
    df_ranks = df[available_metrics].rank(ascending=False)
    df["Avg Rank"] = df_ranks.mean(axis=1)
    overall_best_idx = df["Avg Rank"].idxmin()
    overall_best = df.loc[overall_best_idx, "Model"]
    print(f"\n  🏆 OVERALL BEST (by avg rank): {overall_best}")

# ============================================================
# 8. SAVE SUMMARY REPORT
# ============================================================
summary = {
    "dataset": DATASET_NAME,
    "num_models_compared": len(df),
    "models": list(df["Model"]),
    "metrics_compared": available_metrics,
    "best_per_metric": best_models,
    "overall_best": overall_best if len(available_metrics) > 1 else list(best_models.values())[0]["model"],
    "comparison_table": df.to_dict(orient="records")
}

summary_path = os.path.join(OUTPUT_DIR, "comparison_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\n✓ Summary report saved: {summary_path}")

# ============================================================
# 9. GENERATE MARKDOWN REPORT
# ============================================================
md_report = f"""# Model Comparison Report
## Dataset: {DATASET_NAME}

### Summary
- **Models Compared**: {len(df)}
- **Best Overall Model**: {overall_best if len(available_metrics) > 1 else list(best_models.values())[0]["model"]}

### Comparison Table

{df.to_markdown(index=False)}

### Best Model per Metric

| Metric | Best Model | Score |
|--------|------------|-------|
"""

for metric, info in best_models.items():
    md_report += f"| {metric} | {info['model']} | {info['score']:.4f} |\n"

md_report += """
### Visualizations

![Bar Chart](model_comparison_bar.png)

![Heatmap](model_comparison_heatmap.png)

![Radar Chart](model_comparison_radar.png)

---
*Generated automatically by compare_models.py*
"""

md_path = os.path.join(OUTPUT_DIR, "comparison_report.md")
with open(md_path, 'w') as f:
    f.write(md_report)
print(f"✓ Markdown report saved: {md_path}")

print(f"\n{'='*60}")
print("MODEL COMPARISON COMPLETE!")
print(f"{'='*60}")
print(f"\nResults saved to: {OUTPUT_DIR}/")
print("  - model_comparison.csv")
print("  - comparison_summary.json")
print("  - comparison_report.md")
print("  - model_comparison_*.png (charts)")
