#!/usr/bin/env python3
"""
Unified Model Evaluation Script

Evaluates all trained models with:
- Consistent preprocessing
- Comprehensive metrics (including Hamming accuracy)
- Standardized output format
- Diagnostic reports
"""
#evaluate_all_models.py
import os
import sys
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from improved_metrics import comprehensive_metrics, print_metrics_summary

# Configuration
DATASET_CONFIG = {
    "name": "go_emotions",
    "subset": "simplified",
    "text_column": "text",
    "label_column": "labels",
}

# Model paths - UPDATE THESE  AS NEEDED
MODEL_PATHS = {
    "baseline": "./outputs/go-emotions/final_model",
    "kfold": "./kfold_model_go-emotions/final_distilbert_model_kfold",
    "weighted": "./outputs/weighted_model_go-emotions/final_model",
    "partial_freezing": "./layer_analysis_go-emotions/final_model",
    "mlm": "./outputs/continued_pretraining_go-emotions/final_model",
    "supcon": "./outputs/go-emotions/supcon_pretraining",
}

OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.5


def setup_output_dirs(model_name: str) -> Path:
    """Create standardized output directory structure."""
    model_dir = OUTPUT_DIR / model_name
    (model_dir / "analysis").mkdir(parents=True, exist_ok=True)
    return model_dir


def load_test_dataset():
    """Load and prepare test dataset."""
    print("Loading test dataset...")
    dataset = load_dataset(DATASET_CONFIG["name"], DATASET_CONFIG["subset"])
    test_dataset = dataset["test"]
    train_dataset = dataset["train"]
    
    # Get label info
    label_names = train_dataset.features[DATASET_CONFIG["label_column"]].feature.names
    num_labels = len(label_names)
    
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Labels: {num_labels}")
    print(f"  Label names: {label_names[:5]}... (showing first 5)")
    
    return test_dataset, label_names, num_labels


def tokenize_dataset(test_dataset, tokenizer):
    """Tokenize test dataset."""
    print("Tokenizing...")
    
    def tokenize_function(examples):
        return tokenizer(examples[DATASET_CONFIG["text_column"]], 
                        truncation=True, 
                        max_length=128,
                        padding="max_length")
    
    tokenized = test_dataset.map(tokenize_function, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
    return tokenized


def convert_labels_to_binary(raw_labels, num_labels):
    """Convert list of label indices to binary matrix  ."""
    print("Converting labels to binary format...")
    true_labels = np.zeros((len(raw_labels), num_labels), dtype=int)
    for i, label_indices in enumerate(raw_labels):
        for idx in label_indices:
            if 0 <= idx < num_labels:
                true_labels[i, idx] = 1
    return true_labels


def evaluate_model(model_path: str, model_name: str, 
                   tokenized_test, true_labels, label_names):
    """Evaluate a single model and save results."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Setup output directory
    output_dir = setup_output_dirs(model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        return None
    
    try:
        # Load model - handle PEFT/LoRA models
        print(f"Loading model from {model_path}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"⚠️  LoRA/PEFT adapter mismatch detected. Skipping {model_name}.")
                print(f"    Error: {str(e)[:200]}...")
                return None
            else:
                raise
        
        model.eval()
        model.to(device)
        
        # Get predictions
        print("Generating predictions...")
        all_logits = []
        all_probs = []
        
        batch_size = 32
        for i in range(0, len(tokenized_test), batch_size):
            batch = tokenized_test[i:i+batch_size]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                
                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        logits_array = np.vstack(all_logits)
        probs_array = np.vstack(all_probs)
        predictions_array = (probs_array > THRESHOLD).astype(int)
        
        print(f"  Predictions shape: {predictions_array.shape}")
        print(f"  Probabilities shape: {probs_array.shape}")
        
        # Calculate comprehensive metrics
        print("Calculating metrics...")
        metrics = comprehensive_metrics(
            y_true=true_labels,
            y_pred=predictions_array,
            y_prob=probs_array,
            label_names=label_names,
            threshold=THRESHOLD
        )
        
        # Print summary
        print_metrics_summary(metrics, show_per_label=False)
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to: {metrics_path}")
        
        # Save predictions
        predictions_path = output_dir / "predictions.npz"
        np.savez_compressed(
            predictions_path,
            predictions=predictions_array,
            probabilities=probs_array,
            true_labels=true_labels
        )
        print(f"✓ Predictions saved to: {predictions_path}")
        
        # Create visualizations
        create_visualizations(predictions_array, probs_array, true_labels, 
                            label_names, output_dir)
        
        return metrics
        
    except Exception as e:
        print(f"✗ Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_visualizations(predictions, probabilities, true_labels, 
                         label_names, output_dir):
    """Create diagnostic visualizations."""
    print("Creating visualizations...")
    
    # 1. Prediction distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    preds_per_sample = predictions.sum(axis=1)
    true_per_sample = true_labels.sum(axis=1)
    
    ax1.hist(preds_per_sample, bins=range(0, max(preds_per_sample.max(), 6)+2), 
             alpha=0.7, label='Predicted', edgecolor='black')
    ax1.hist(true_per_sample, bins=range(0, max(true_per_sample.max(), 6)+2), 
             alpha=0.7, label='True', edgecolor='black')
    ax1.set_xlabel('Number of Labels')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Labels per Sample Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Per-label prediction frequency
    pred_counts = predictions.sum(axis=0)
    true_counts = true_labels.sum(axis=0)
    
    x = np.arange(len(label_names))
    width = 0.35
    
    ax2.bar(x - width/2, true_counts, width, label='True', alpha=0.7)
    ax2.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Per-Label Frequency')
    ax2.set_xticks(x)
    ax2.set_xticklabels(label_names, rotation=90, ha='right', fontsize=6)
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis" / "prediction_distribution.png", dpi=150)
    plt.close()
    
    # 3. Probability heatmap (sample)
    fig, ax = plt.subplots(figsize=(14, 10))
    sample_size = min(100, probabilities.shape[0])
    sample_probs = probabilities[:sample_size, :]
    sns.heatmap(sample_probs.T, cmap='RdYlGn', vmin=0, vmax=1, 
                yticklabels=label_names, xticklabels=False,
                cbar_kws={'label': 'Probability'})
    ax.set_ylabel('Labels')
    ax.set_xlabel(f'Test Samples (first {sample_size})')
    ax.set_title('Prediction Probabilities Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / "analysis" / "probability_heatmap.png", dpi=150)
    plt.close()
    
    print("✓ Visualizations saved")


def create_comparison_table(all_metrics):
    """Create comparison table across all models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if not all_metrics:
        print("No metrics to compare")
        return
    
    # Key metrics to compare
    key_metrics = [
        "hamming_accuracy",
        "f1_micro",
        "f1_weighted",
        "subset_accuracy",
        "precision_weighted",
        "recall_weighted"
    ]
    
    print(f"\n{'Model':<20} ", end="")
    for metric in key_metrics:
        print(f"{metric.replace('_', ' ').title():<18} ", end="")
    print()
    print("-" * 130)
    
    for model_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        print(f"{model_name:<20} ", end="")
        for metric in key_metrics:
            value = metrics.get(metric, 0.0)
            print(f"{value:<18.4f} ", end="")
        print()
    
    # Ranking by Hamming accuracy
    print(f"\n{'='*60}")
    print("RANKING BY HAMMING ACCURACY (Recommended for Multi-Label)")
    print(f"{'='*60}")
    
    ranked = sorted(
        [(name, m.get("hamming_accuracy", 0)) for name, m in all_metrics.items() if m],
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (name, score) in enumerate(ranked, 1):
        print(f"{i}. {name:<20} {score:.4f}")
    
    # Save comparison
    comparison_data = {}
    for model_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        comparison_data[model_name] = {
            metric: metrics.get(metric, 0.0)
            for metric in key_metrics
        }
    
    with open(OUTPUT_DIR / "model_comparison.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\n✓ Comparison saved to: {OUTPUT_DIR / 'model_comparison.json'}")


def main():
    print(f"\n{'='*70}")
    print("UNIFIED MODEL EVALUATION")
    print(f"{'='*70}")
    
    # Load dataset
    test_dataset, label_names, num_labels = load_test_dataset()
    
    # Convert labels
    raw_labels = test_dataset[DATASET_CONFIG["label_column"]]
    true_labels = convert_labels_to_binary(raw_labels, num_labels)
    print(f"  True labels shape: {true_labels.shape}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenize dataset
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)
    
    # Evaluate all models
    all_metrics = {}
    for model_name, model_path in MODEL_PATHS.items():
        metrics = evaluate_model(
            model_path=model_path,
            model_name=model_name,
            tokenized_test=tokenized_test,
            true_labels=true_labels,
            label_names=label_names
        )
        all_metrics[model_name] = metrics
    
    # Create comparison
    create_comparison_table(all_metrics)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nDirectory structure:")
    print("  results/")
    print("    ├── model_comparison.json")
    print("    ├── <model_name>/")
    print("    │   ├── metrics.json")
    print("    │   ├── predictions.npz")
    print("    │   └── analysis/")
    print("    │       ├── prediction_distribution.png")
    print("    │       └── probability_heatmap.png")


if __name__ == "__main__":
    main()
