#improved_metrics.py
"""
Improved Metrics Module for Multi-Label Emotion Classification

Provides comprehensive evaluation metrics beyond standard F1/accuracy:
- Hamming accuracy (per-label average)
- Per-label precision/recall/F1
- Prediction distribution statistics
- Probability calibration metrics
- Subset accuracy (with appropriate warnings)
- Coverage and ranking metrics
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    coverage_error,
    label_ranking_average_precision_score
)
from typing import Dict, List, Tuple, Optional
import warnings


def hamming_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Hamming accuracy (complement of Hamming loss).
    
    This is the average per-label accuracy - the fraction of labels 
    that are correctly predicted across all samples.
    
    Args:
        y_true: True labels, shape (n_samples, n_labels)
        y_pred: Predicted labels, shape (n_samples, n_labels)
    
    Returns:
        Hamming accuracy score (0-1)
    """
    return 1 - hamming_loss(y_true, y_pred)


def subset_accuracy_with_warning(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, str]:
    """
    Calculate subset accuracy and return with interpretation warning.
    
    Subset accuracy = fraction of samples where ALL labels are correct.
    This metric can be misleadingly high for sparse multi-label problems.
    
    Args:
        y_true: True labels, shape (n_samples, n_labels)
        y_pred: Predicted labels, shape (n_samples, n_labels)
    
    Returns:
        (accuracy, warning_message)
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate sparsity
    avg_labels = y_true.sum() / (y_true.shape[0] * y_true.shape[1])
    sparsity_pct = (1 - avg_labels) * 100
    
    if sparsity_pct > 90:
        warning = (f"WARNING: Subset accuracy may be inflated due to high sparsity "
                  f"({sparsity_pct:.1f}% of labels are 0). "
                  f"Consider Hamming accuracy as primary metric.")
    else:
        warning = ""
    
    return acc, warning


def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     label_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, F1 for each label individually.
    
    Args:
        y_true: True labels, shape (n_samples, n_labels)
        y_pred: Predicted labels, shape (n_samples, n_labels)
        label_names: List of label names
    
    Returns:
        Dictionary mapping label names to their metrics
    """
    n_labels = y_true.shape[1]
    per_label_stats = {}
    
    for i, label_name in enumerate(label_names):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        
        # Calculate metrics
        precision = precision_score(y_true_label, y_pred_label, zero_division=0)
        recall = recall_score(y_true_label, y_pred_label, zero_division=0)
        f1 = f1_score(y_true_label, y_pred_label, zero_division=0)
        
        # Support (number of true instances)
        support = int(y_true_label.sum())
        
        # Prediction count
        pred_count = int(y_pred_label.sum())
        
        per_label_stats[label_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "predicted_count": pred_count,
            "prediction_rate": float(pred_count / len(y_pred_label)),
            "true_rate": float(support / len(y_true_label))
        }
    
    return per_label_stats


def prediction_distribution_stats(y_pred: np.ndarray, 
                                  y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Analyze the distribution of predictions.
    
    Args:
        y_pred: Predicted labels, shape (n_samples, n_labels)
        y_true: Optional true labels for comparison
    
    Returns:
        Dictionary of distribution statistics
    """
    # Predictions per sample
    preds_per_sample = y_pred.sum(axis=1)
    
    stats = {
        "total_predictions": int(y_pred.sum()),
        "total_possible": int(y_pred.size),
        "positive_rate": float(y_pred.sum() / y_pred.size),
        "predictions_per_sample_mean": float(preds_per_sample.mean()),
        "predictions_per_sample_median": float(np.median(preds_per_sample)),
        "predictions_per_sample_std": float(preds_per_sample.std()),
        "predictions_per_sample_min": int(preds_per_sample.min()),
        "predictions_per_sample_max": int(preds_per_sample.max()),
        "samples_with_zero_predictions": int((preds_per_sample == 0).sum()),
        "samples_with_zero_predictions_pct": float((preds_per_sample == 0).sum() / len(preds_per_sample) * 100)
    }
    
    if y_true is not None:
        true_per_sample = y_true.sum(axis=1)
        stats["true_labels_per_sample_mean"] = float(true_per_sample.mean())
        stats["true_labels_per_sample_median"] = float(np.median(true_per_sample))
        stats["prediction_bias"] = float(preds_per_sample.mean() - true_per_sample.mean())
    
    return stats


def probability_calibration_stats(y_prob: np.ndarray, 
                                  threshold: float = 0.5) -> Dict[str, float]:
    """
    Analyze probability distribution and calibration.
    
    Args:
        y_prob: Predicted probabilities, shape (n_samples, n_labels)
        threshold: Classification threshold
    
    Returns:
        Dictionary of calibration statistics
    """
    stats = {
        "prob_mean": float(y_prob.mean()),
        "prob_median": float(np.median(y_prob)),
        "prob_std": float(y_prob.std()),
        "prob_min": float(y_prob.min()),
        "prob_max": float(y_prob.max()),
        "prob_above_threshold_pct": float((y_prob > threshold).sum() / y_prob.size * 100),
        "prob_very_confident_high_pct": float((y_prob > 0.9).sum() / y_prob.size * 100),
        "prob_very_confident_low_pct": float((y_prob < 0.1).sum() / y_prob.size * 100),
        "prob_near_threshold_pct": float((np.abs(y_prob - threshold) < 0.1).sum() / y_prob.size * 100)
    }
    
    return stats


def comprehensive_metrics(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          y_prob: Optional[np.ndarray] = None,
                          label_names: Optional[List[str]] = None,
                          threshold: float = 0.5) -> Dict:
    """
    Calculate all comprehensive metrics for multi-label classification.
    
    Args:
        y_true: True labels, shape (n_samples, n_labels)
        y_pred: Predicted binary labels, shape (n_samples, n_labels)
        y_prob: Optional predicted probabilities
        label_names: Optional list of label names
        threshold: Classification threshold (default 0.5)
    
    Returns:
        Comprehensive metrics dictionary
    """
    if label_names is None:
        label_names = [f"label_{i}" for i in range(y_true.shape[1])]
    
    # Standard metrics
    metrics = {
        # Core metrics
        "hamming_accuracy": float(hamming_accuracy(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    
    # Subset accuracy with warning
    subset_acc, warning = subset_accuracy_with_warning(y_true, y_pred)
    metrics["subset_accuracy"] = float(subset_acc)
    if warning:
        metrics["subset_accuracy_warning"] = warning
    
    # Ranking metrics (require probabilities)
    if y_prob is not None:
        try:
            metrics["coverage_error"] = float(coverage_error(y_true, y_prob))
            metrics["label_ranking_avg_precision"] = float(
                label_ranking_average_precision_score(y_true, y_prob)
            )
        except Exception as e:
            warnings.warn(f"Could not calculate ranking metrics: {e}")
    
    # Per-label metrics
    metrics["per_label"] = per_label_metrics(y_true, y_pred, label_names)
    
    # Distribution statistics
    metrics["prediction_distribution"] = prediction_distribution_stats(y_pred, y_true)
    
    # Probability calibration (if available)
    if y_prob is not None:
        metrics["probability_calibration"] = probability_calibration_stats(y_prob, threshold)
    
    return metrics


def print_metrics_summary(metrics: Dict, show_per_label: bool = False):
    """
    Print a human-readable summary of metrics.
    
    Args:
        metrics: Metrics dictionary from comprehensive_metrics()
        show_per_label: Whether to print per-label metrics
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("="*60)
    
    print("\n--- Primary Metrics ---")
    print(f"Hamming Accuracy:     {metrics['hamming_accuracy']:.4f} ← RECOMMENDED for multi-label")
    print(f"F1-Micro:             {metrics['f1_micro']:.4f}")
    print(f"F1-Weighted:          {metrics['f1_weighted']:.4f}")
    print(f"F1-Macro:             {metrics['f1_macro']:.4f}")
    
    print("\n--- Subset Accuracy (All Labels Correct) ---")
    print(f"Subset Accuracy:      {metrics['subset_accuracy']:.4f}")
    if "subset_accuracy_warning" in metrics:
        print(f"⚠️  {metrics['subset_accuracy_warning']}")
    
    print("\n--- Precision/Recall ---")
    print(f"Precision (Micro):    {metrics['precision_micro']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Micro):       {metrics['recall_micro']:.4f}")
    print(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}")
    
    if "prediction_distribution" in metrics:
        dist = metrics["prediction_distribution"]
        print("\n--- Prediction Distribution ---")
        print(f"Total predictions:    {dist['total_predictions']}/{dist['total_possible']} ({dist['positive_rate']*100:.2f}%)")
        print(f"Predictions/sample:   mean={dist['predictions_per_sample_mean']:.2f}, "
              f"median={dist['predictions_per_sample_median']:.0f}, "
              f"range=[{dist['predictions_per_sample_min']}, {dist['predictions_per_sample_max']}]")
        print(f"Zero predictions:     {dist['samples_with_zero_predictions']} samples "
              f"({dist['samples_with_zero_predictions_pct']:.1f}%)")
        if "prediction_bias" in dist:
            print(f"Prediction bias:      {dist['prediction_bias']:+.2f} (pred - true)")
    
    if "probability_calibration" in metrics:
        prob = metrics["probability_calibration"]
        print("\n--- Probability Calibration ---")
        print(f"Mean probability:     {prob['prob_mean']:.4f}")
        print(f"Prob > 0.9:           {prob['prob_very_confident_high_pct']:.2f}%")
        print(f"Prob < 0.1:           {prob['prob_very_confident_low_pct']:.2f}%")
    
    if show_per_label and "per_label" in metrics:
        print("\n--- Per-Label Performance (Top 10 by F1) ---")
        per_label = metrics["per_label"]
        sorted_labels = sorted(per_label.items(), 
                             key=lambda x: x[1]["f1"], 
                             reverse=True)[:10]
        
        print(f"{'Label':<20} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Support':<10} {'Pred Count':<12}")
        print("-" * 80)
        for label, stats in sorted_labels:
            print(f"{label:<20} {stats['f1']:<8.4f} {stats['precision']:<8.4f} "
                  f"{stats['recall']:<8.4f} {stats['support']:<10} {stats['predicted_count']:<12}")
    
    print("\n" + "="*60)
