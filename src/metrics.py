# metrics.py
"""
Metrics Module for Multi-Label / Single-Label Emotion Classification.

Provides:
- compute_metrics(): Trainer-compatible metric function
- make_compute_metrics(): Convenience wrapper factory
- comprehensive_metrics(): Full evaluation with per-label stats
- print_metrics_summary(): Human-readable output
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    coverage_error,
    label_ranking_average_precision_score,
)
from typing import Dict, List, Optional, Tuple
import warnings


# ──────────────────────────────────────────────────────────────
# Trainer-compatible metric function
# ──────────────────────────────────────────────────────────────

def compute_metrics(eval_pred, is_multilabel: bool = False) -> Dict[str, float]:
    """
    Compute metrics for the HuggingFace Trainer callback.

    Args:
        eval_pred: (logits, labels) tuple from Trainer.
        is_multilabel: Whether the task is multi-label.

    Returns:
        Dictionary of scalar metrics.
    """
    logits, labels = eval_pred

    if is_multilabel:
        probs = 1.0 / (1.0 + np.exp(-logits))  # numpy sigmoid
        predictions = (probs > 0.5).astype(int)
        labels = labels.astype(int)

        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "hamming_accuracy": float(1 - hamming_loss(labels, predictions)),
            "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            "f1_micro": float(f1_score(labels, predictions, average="micro", zero_division=0)),
            "precision_weighted": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        }
    else:
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            "precision_weighted": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        }


def make_compute_metrics(is_multilabel: bool):
    """Return a zero-argument-compatible wrapper for Trainer's compute_metrics."""
    def _fn(eval_pred):
        return compute_metrics(eval_pred, is_multilabel=is_multilabel)
    return _fn


# ──────────────────────────────────────────────────────────────
# Comprehensive evaluation helpers
# ──────────────────────────────────────────────────────────────

def hamming_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hamming accuracy = 1 - hamming_loss."""
    return 1 - hamming_loss(y_true, y_pred)


def subset_accuracy_with_warning(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, str]:
    """Subset accuracy with sparsity warning."""
    acc = accuracy_score(y_true, y_pred)
    avg_labels = y_true.sum() / (y_true.shape[0] * y_true.shape[1])
    sparsity_pct = (1 - avg_labels) * 100
    warning = ""
    if sparsity_pct > 90:
        warning = (
            f"WARNING: Subset accuracy may be inflated due to high sparsity "
            f"({sparsity_pct:.1f}% of labels are 0). "
            f"Consider Hamming accuracy as primary metric."
        )
    return acc, warning


def per_label_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Per-label precision / recall / F1 / support."""
    per_label_stats = {}
    for i, label_name in enumerate(label_names):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        support = int(y_true_label.sum())
        pred_count = int(y_pred_label.sum())
        per_label_stats[label_name] = {
            "precision": float(precision_score(y_true_label, y_pred_label, zero_division=0)),
            "recall": float(recall_score(y_true_label, y_pred_label, zero_division=0)),
            "f1": float(f1_score(y_true_label, y_pred_label, zero_division=0)),
            "support": support,
            "predicted_count": pred_count,
            "prediction_rate": float(pred_count / len(y_pred_label)),
            "true_rate": float(support / len(y_true_label)),
        }
    return per_label_stats


def prediction_distribution_stats(
    y_pred: np.ndarray, y_true: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Analyse the distribution of predictions."""
    pps = y_pred.sum(axis=1)
    stats: Dict[str, float] = {
        "total_predictions": int(y_pred.sum()),
        "total_possible": int(y_pred.size),
        "positive_rate": float(y_pred.sum() / y_pred.size),
        "predictions_per_sample_mean": float(pps.mean()),
        "predictions_per_sample_median": float(np.median(pps)),
        "predictions_per_sample_std": float(pps.std()),
        "predictions_per_sample_min": int(pps.min()),
        "predictions_per_sample_max": int(pps.max()),
        "samples_with_zero_predictions": int((pps == 0).sum()),
        "samples_with_zero_predictions_pct": float((pps == 0).sum() / len(pps) * 100),
    }
    if y_true is not None:
        tps = y_true.sum(axis=1)
        stats["true_labels_per_sample_mean"] = float(tps.mean())
        stats["true_labels_per_sample_median"] = float(np.median(tps))
        stats["prediction_bias"] = float(pps.mean() - tps.mean())
    return stats


def probability_calibration_stats(
    y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Analyse probability distribution and calibration."""
    return {
        "prob_mean": float(y_prob.mean()),
        "prob_median": float(np.median(y_prob)),
        "prob_std": float(y_prob.std()),
        "prob_min": float(y_prob.min()),
        "prob_max": float(y_prob.max()),
        "prob_above_threshold_pct": float((y_prob > threshold).sum() / y_prob.size * 100),
        "prob_very_confident_high_pct": float((y_prob > 0.9).sum() / y_prob.size * 100),
        "prob_very_confident_low_pct": float((y_prob < 0.1).sum() / y_prob.size * 100),
        "prob_near_threshold_pct": float(
            (np.abs(y_prob - threshold) < 0.1).sum() / y_prob.size * 100
        ),
    }


def comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict:
    """All-in-one comprehensive metrics for multi-label classification."""
    if label_names is None:
        label_names = [f"label_{i}" for i in range(y_true.shape[1])]

    metrics: Dict = {
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

    subset_acc, warning = subset_accuracy_with_warning(y_true, y_pred)
    metrics["subset_accuracy"] = float(subset_acc)
    if warning:
        metrics["subset_accuracy_warning"] = warning

    if y_prob is not None:
        try:
            metrics["coverage_error"] = float(coverage_error(y_true, y_prob))
            metrics["label_ranking_avg_precision"] = float(
                label_ranking_average_precision_score(y_true, y_prob)
            )
        except Exception as e:
            warnings.warn(f"Could not calculate ranking metrics: {e}")

    metrics["per_label"] = per_label_metrics(y_true, y_pred, label_names)
    metrics["prediction_distribution"] = prediction_distribution_stats(y_pred, y_true)

    if y_prob is not None:
        metrics["probability_calibration"] = probability_calibration_stats(y_prob, threshold)

    return metrics


def print_metrics_summary(metrics: Dict, show_per_label: bool = False):
    """Print a human-readable summary of comprehensive metrics."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("=" * 60)

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
        print(
            f"Total predictions:    {dist['total_predictions']}/{dist['total_possible']} "
            f"({dist['positive_rate']*100:.2f}%)"
        )
        print(
            f"Predictions/sample:   mean={dist['predictions_per_sample_mean']:.2f}, "
            f"median={dist['predictions_per_sample_median']:.0f}, "
            f"range=[{dist['predictions_per_sample_min']}, {dist['predictions_per_sample_max']}]"
        )
        print(
            f"Zero predictions:     {dist['samples_with_zero_predictions']} samples "
            f"({dist['samples_with_zero_predictions_pct']:.1f}%)"
        )
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
        sorted_labels = sorted(
            metrics["per_label"].items(), key=lambda x: x[1]["f1"], reverse=True
        )[:10]
        print(f"{'Label':<20} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Support':<10} {'Pred Count':<12}")
        print("-" * 80)
        for label, stats in sorted_labels:
            print(
                f"{label:<20} {stats['f1']:<8.4f} {stats['precision']:<8.4f} "
                f"{stats['recall']:<8.4f} {stats['support']:<10} {stats['predicted_count']:<12}"
            )

    print("\n" + "=" * 60)
