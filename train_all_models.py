#!/usr/bin/env python3
"""
Unified Training Pipeline for DistilBERT Emotion Classification

Trains all model variants with consistent configuration:
1. Baseline - Standard fine-tuning
2. K-Fold - Cross-validation with final model on full train+val
3. Weighted - Class-weighted loss for imbalanced classes
4. Partial Freezing - Freeze lower layers, train only top N layers

All models:
- Use full fine-tuning (no LoRA)
- Use 28-class go_emotions (simplified + neutral)
- Save to standardized results/ directory
- Include validation after saving
"""
#train_all_models.py

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from improved_metrics import comprehensive_metrics, print_metrics_summary

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_CONFIG = {
    "name": "go_emotions",
    "subset": "simplified",
    "text_column": "text",
    "label_column": "labels",
}

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "logging_steps": 100,
    "seed": 42,
}

# Models to train
MODELS_TO_TRAIN = ["baseline", "kfold", "weighted", "partial_freezing"]


# =============================================================================
# DATA COLLATOR FOR MULTI-LABEL
# =============================================================================
@dataclass
class MultiLabelDataCollator:
    """Data collator that handles multi-label classification."""
    tokenizer: Any
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [feature.pop("labels") for feature in features] if "labels" in features[0] else None
        batch = self.tokenizer.pad(features, padding=self.padding, return_tensors="pt")
        
        if labels is not None:
            if isinstance(labels[0], torch.Tensor):
                batch["labels"] = torch.stack(labels).float()
            elif isinstance(labels[0], list):
                batch["labels"] = torch.tensor(labels, dtype=torch.float)
            else:
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(eval_pred):
    """Compute metrics for multi-label classification."""
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_micro": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
    }


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load and prepare dataset."""
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    dataset = load_dataset(DATASET_CONFIG["name"], DATASET_CONFIG["subset"])
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    
    # Get label info
    label_names = train_dataset.features[DATASET_CONFIG["label_column"]].feature.names
    num_labels = len(label_names)
    
    print(f"Dataset: {DATASET_CONFIG['name']} ({DATASET_CONFIG['subset']})")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Labels: {num_labels}")
    
    return dataset, label_names, num_labels


def tokenize_and_prepare(dataset, tokenizer, num_labels):
    """Tokenize dataset and convert labels to binary format."""
    
    def tokenize_function(examples):
        return tokenizer(
            examples[DATASET_CONFIG["text_column"]],
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    def convert_labels(examples):
        """Convert label indices to binary multi-hot vectors."""
        batch_labels = []
        for label_indices in examples[DATASET_CONFIG["label_column"]]:
            binary_labels = [0] * num_labels
            for idx in label_indices:
                if 0 <= idx < num_labels:
                    binary_labels[idx] = 1
            batch_labels.append(binary_labels)
        return {"labels": batch_labels}
    
    # Tokenize
    tokenized = dataset.map(tokenize_function, batched=True)
    
    # Convert labels
    tokenized = tokenized.map(convert_labels, batched=True)
    
    # Remove text column
    if DATASET_CONFIG["text_column"] in tokenized.column_names:
        tokenized = tokenized.remove_columns([DATASET_CONFIG["text_column"]])
    
    # Remove original label column if different from "labels"
    if DATASET_CONFIG["label_column"] != "labels" and DATASET_CONFIG["label_column"] in tokenized.column_names:
        tokenized = tokenized.remove_columns([DATASET_CONFIG["label_column"]])
    
    # Keep only necessary columns
    keep_cols = ["input_ids", "attention_mask", "labels"]
    remove_cols = [c for c in tokenized.column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(remove_cols)
    
    tokenized.set_format("torch")
    return tokenized


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================
def get_model(num_labels, label_names):
    """Initialize a fresh model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    model.config.id2label = {i: label for i, label in enumerate(label_names)}
    model.config.label2id = {label: i for i, label in enumerate(label_names)}
    return model


def train_baseline(train_dataset, val_dataset, test_dataset, num_labels, label_names, tokenizer):
    """Train baseline model with standard fine-tuning."""
    print("\n" + "="*60)
    print("TRAINING: BASELINE")
    print("="*60)
    
    output_dir = RESULTS_DIR / "baseline"
    output_dir.mkdir(exist_ok=True)
    
    model = get_model(num_labels, label_names)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=TRAINING_CONFIG["learning_rate"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=TRAINING_CONFIG["logging_steps"],
        seed=TRAINING_CONFIG["seed"],
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=MultiLabelDataCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Save model
    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"✓ Model saved to {model_path}")
    
    # Evaluate on test set
    return evaluate_and_save(trainer, test_dataset, output_dir, "baseline", label_names)


def train_kfold(full_train_val_dataset, test_dataset, num_labels, label_names, tokenizer, n_splits=4):
    """Train with K-Fold cross-validation, then final model on full data."""
    print("\n" + "="*60)
    print(f"TRAINING: K-FOLD (K={n_splits})")
    print("="*60)
    
    output_dir = RESULTS_DIR / "kfold"
    output_dir.mkdir(exist_ok=True)
    
    # K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=TRAINING_CONFIG["seed"])
    indices = np.arange(len(full_train_val_dataset))
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        train_fold = full_train_val_dataset.select(train_idx)
        val_fold = full_train_val_dataset.select(val_idx)
        
        model = get_model(num_labels, label_names)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir / f"fold_{fold}"),
            learning_rate=TRAINING_CONFIG["learning_rate"],
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
            num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=TRAINING_CONFIG["logging_steps"],
            seed=TRAINING_CONFIG["seed"],
            report_to="none",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            processing_class=tokenizer,
            data_collator=MultiLabelDataCollator(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        fold_result = trainer.evaluate()
        fold_metrics.append(fold_result)
        print(f"Fold {fold + 1}: F1-Weighted = {fold_result['eval_f1_weighted']:.4f}")
    
    # Save fold metrics
    avg_f1 = np.mean([m['eval_f1_weighted'] for m in fold_metrics])
    std_f1 = np.std([m['eval_f1_weighted'] for m in fold_metrics])
    print(f"\nK-Fold CV: F1-Weighted = {avg_f1:.4f} ± {std_f1:.4f}")
    
    with open(output_dir / "fold_metrics.json", 'w') as f:
        json.dump({
            "n_splits": n_splits,
            "fold_results": fold_metrics,
            "average_f1_weighted": float(avg_f1),
            "std_f1_weighted": float(std_f1)
        }, f, indent=2)
    
    # Train final model on full train+val
    print("\n--- Training Final Model on Full Train+Val ---")
    model = get_model(num_labels, label_names)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "final_training"),
        learning_rate=TRAINING_CONFIG["learning_rate"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        eval_strategy="no",
        save_strategy="no",
        logging_steps=TRAINING_CONFIG["logging_steps"],
        seed=TRAINING_CONFIG["seed"],
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_val_dataset,
        processing_class=tokenizer,
        data_collator=MultiLabelDataCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Save final model
    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"✓ Model saved to {model_path}")
    
    return evaluate_and_save(trainer, test_dataset, output_dir, "kfold", label_names)


def train_weighted(train_dataset, val_dataset, test_dataset, num_labels, label_names, tokenizer):
    """Train with class-weighted loss for imbalanced classes."""
    print("\n" + "="*60)
    print("TRAINING: WEIGHTED (Class-Balanced Loss)")
    print("="*60)
    
    output_dir = RESULTS_DIR / "weighted"
    output_dir.mkdir(exist_ok=True)
    
    # Calculate class weights (pos_weight for BCEWithLogitsLoss)
    print("Calculating class weights...")
    all_labels = np.array([sample["labels"].numpy() for sample in train_dataset])
    pos_counts = all_labels.sum(axis=0)
    neg_counts = len(all_labels) - pos_counts
    
    # pos_weight = neg_count / pos_count (clamped)
    pos_weight = np.clip(neg_counts / (pos_counts + 1e-6), 1.0, 10.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float)
    
    print(f"  Min pos_weight: {pos_weight.min():.2f}")
    print(f"  Max pos_weight: {pos_weight.max():.2f}")
    
    # Custom loss function
    class WeightedBCELoss(torch.nn.Module):
        def __init__(self, pos_weight):
            super().__init__()
            self.register_buffer("pos_weight", pos_weight)
            
        def forward(self, logits, labels):
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_weight
            )
    
    # Custom Trainer with weighted loss
    class WeightedTrainer(Trainer):
        def __init__(self, pos_weight, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_fn = WeightedBCELoss(pos_weight)
            
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Move pos_weight to correct device
            if self.loss_fn.pos_weight.device != logits.device:
                self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)
            
            loss = self.loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    model = get_model(num_labels, label_names)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=TRAINING_CONFIG["learning_rate"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=TRAINING_CONFIG["logging_steps"],
        seed=TRAINING_CONFIG["seed"],
        report_to="none",
    )
    
    trainer = WeightedTrainer(
        pos_weight=pos_weight_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=MultiLabelDataCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Save model
    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    
    # Save class weights
    with open(output_dir / "class_weights.json", 'w') as f:
        json.dump({label: float(w) for label, w in zip(label_names, pos_weight)}, f, indent=2)
    
    print(f"✓ Model saved to {model_path}")
    
    return evaluate_and_save(trainer, test_dataset, output_dir, "weighted", label_names)


def train_partial_freezing(train_dataset, val_dataset, test_dataset, num_labels, label_names, tokenizer, layers_unfrozen=3):
    """Train with partial layer freezing (only top N layers trainable)."""
    print("\n" + "="*60)
    print(f"TRAINING: PARTIAL FREEZING ({layers_unfrozen} layers unfrozen)")
    print("="*60)
    
    output_dir = RESULTS_DIR / "partial_freezing"
    output_dir.mkdir(exist_ok=True)
    
    model = get_model(num_labels, label_names)
    
    # Freeze all transformer layers first
    for param in model.distilbert.parameters():
        param.requires_grad = False
    
    # Unfreeze top N layers
    total_layers = len(model.distilbert.transformer.layer)
    for i in range(total_layers - layers_unfrozen, total_layers):
        for param in model.distilbert.transformer.layer[i].parameters():
            param.requires_grad = True
    
    # Always unfreeze classifier
    for param in model.pre_classifier.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=TRAINING_CONFIG["learning_rate"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=TRAINING_CONFIG["logging_steps"],
        seed=TRAINING_CONFIG["seed"],
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=MultiLabelDataCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Save model
    model_path = output_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"✓ Model saved to {model_path}")
    
    # Save freezing config
    with open(output_dir / "freezing_config.json", 'w') as f:
        json.dump({
            "layers_unfrozen": layers_unfrozen,
            "total_layers": total_layers,
            "trainable_params": trainable,
            "total_params": total
        }, f, indent=2)
    
    return evaluate_and_save(trainer, test_dataset, output_dir, "partial_freezing", label_names)


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_and_save(trainer, test_dataset, output_dir, model_name, label_names):
    """Evaluate model and save comprehensive metrics."""
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    
    # Get predictions
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    true_labels = predictions_output.label_ids
    
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    
    # Calculate comprehensive metrics
    metrics = comprehensive_metrics(
        y_true=true_labels.astype(int),
        y_pred=preds,
        y_prob=probs,
        label_names=label_names,
        threshold=0.5
    )
    
    # Print summary
    print_metrics_summary(metrics, show_per_label=False)
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    np.savez_compressed(
        output_dir / "predictions.npz",
        predictions=preds,
        probabilities=probs,
        true_labels=true_labels
    )
    
    print(f"✓ Metrics and predictions saved to {output_dir}")
    return metrics


def validate_saved_model(model_path, test_dataset, label_names, tokenizer):
    """Validate that a saved model works correctly when loaded."""
    print(f"\n--- Validating Saved Model: {model_path} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(device)
    
    # Quick prediction check
    batch = test_dataset[:32]
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits)
    
    # Check prediction distribution
    mean_prob = probs.mean().item()
    positive_rate = (probs > 0.5).float().mean().item()
    
    print(f"  Mean probability: {mean_prob:.4f}")
    print(f"  Positive rate: {positive_rate*100:.2f}%")
    
    if mean_prob < 0.01:
        print("  ⚠️  WARNING: Model may be predicting too conservatively!")
        return False
    elif positive_rate < 0.01:
        print("  ⚠️  WARNING: Model is predicting almost no positives!")
        return False
    else:
        print("  ✓ Model validation passed")
        return True


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*70)
    print("UNIFIED TRAINING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load data
    dataset, label_names, num_labels = load_data()
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = tokenize_and_prepare(dataset["train"], tokenizer, num_labels)
    val_dataset = tokenize_and_prepare(dataset["validation"], tokenizer, num_labels)
    test_dataset = tokenize_and_prepare(dataset["test"], tokenizer, num_labels)
    
    # For K-fold: combine train + val
    full_train_val = concatenate_datasets([train_dataset, val_dataset])
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"  Train+Val (for K-Fold): {len(full_train_val)}")
    
    # Train models
    all_metrics = {}
    
    for model_type in MODELS_TO_TRAIN:
        try:
            if model_type == "baseline":
                metrics = train_baseline(train_dataset, val_dataset, test_dataset, 
                                        num_labels, label_names, tokenizer)
                # Validate saved model
                validate_saved_model(RESULTS_DIR / "baseline" / "model", 
                                    test_dataset, label_names, tokenizer)
                
            elif model_type == "kfold":
                metrics = train_kfold(full_train_val, test_dataset, 
                                     num_labels, label_names, tokenizer)
                validate_saved_model(RESULTS_DIR / "kfold" / "model", 
                                    test_dataset, label_names, tokenizer)
                
            elif model_type == "weighted":
                metrics = train_weighted(train_dataset, val_dataset, test_dataset,
                                        num_labels, label_names, tokenizer)
                validate_saved_model(RESULTS_DIR / "weighted" / "model",
                                    test_dataset, label_names, tokenizer)
                
            elif model_type == "partial_freezing":
                metrics = train_partial_freezing(train_dataset, val_dataset, test_dataset,
                                                num_labels, label_names, tokenizer)
                validate_saved_model(RESULTS_DIR / "partial_freezing" / "model",
                                    test_dataset, label_names, tokenizer)
            
            all_metrics[model_type] = metrics
            
        except Exception as e:
            print(f"\n✗ Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            all_metrics[model_type] = None
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    # Create comparison table
    print(f"\n{'Model':<20} {'Hamming Acc':<12} {'F1-Micro':<10} {'F1-Weighted':<12} {'Subset Acc':<12}")
    print("-" * 70)
    
    for model_name, metrics in all_metrics.items():
        if metrics:
            print(f"{model_name:<20} {metrics['hamming_accuracy']:<12.4f} "
                  f"{metrics['f1_micro']:<10.4f} {metrics['f1_weighted']:<12.4f} "
                  f"{metrics['subset_accuracy']:<12.4f}")
        else:
            print(f"{model_name:<20} {'FAILED':<12}")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": DATASET_CONFIG,
        "training_config": TRAINING_CONFIG,
        "models_trained": MODELS_TO_TRAIN,
        "results": {
            name: {
                "hamming_accuracy": m["hamming_accuracy"],
                "f1_micro": m["f1_micro"],
                "f1_weighted": m["f1_weighted"],
                "f1_macro": m["f1_macro"],
                "subset_accuracy": m["subset_accuracy"]
            } if m else None
            for name, m in all_metrics.items()
        }
    }
    
    with open(RESULTS_DIR / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ All results saved to: {RESULTS_DIR}/")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
