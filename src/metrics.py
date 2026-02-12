#metrics.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred, is_multilabel=False):
    """
    Compute metrics for single or multi-label classification.
    """
    logits, labels = eval_pred
    
    # Convert logits to tensor if numpy array
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    
    if is_multilabel:
        # Multi-label: sigmoid + threshold
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).int().numpy()
        labels = labels.astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
        f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
        precision_weighted = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall_weighted = recall_score(labels, predictions, average="weighted", zero_division=0)
        
        return {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_micro": f1_micro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted
        }
    else:
        # Single-label: argmax
        predictions = np.argmax(logits.numpy(), axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        precision = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(labels, predictions, average="weighted", zero_division=0)
        
        return {
            "accuracy": accuracy,
            "f1_weighted": f1,
            "precision_weighted": precision,
            "recall_weighted": recall
        }
