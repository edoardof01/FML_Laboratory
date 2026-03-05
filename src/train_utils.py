# train_utils.py
"""
Training utilities: data collation, weighted trainer, training-arg builder,
seed setting, and evaluation / save helpers.
"""
import os
import json
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


# ──────────────────────────────────────────────────────────────
# Data Collator
# ──────────────────────────────────────────────────────────────

@dataclass
class MultiLabelDataCollator:
    """Data collator that handles multi-label classification properly."""

    tokenizer: Any
    padding: Union[bool, str] = True
    max_length: int = 128

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = (
            [feature.pop("labels") for feature in features]
            if "labels" in features[0]
            else None
        )

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if labels is not None:
            if isinstance(labels[0], torch.Tensor):
                batch["labels"] = (
                    torch.stack(labels)
                    if labels[0].dim() == 0
                    else torch.stack(labels).float()
                )
            elif isinstance(labels[0], list):
                batch["labels"] = torch.tensor(labels, dtype=torch.float)
            else:
                batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


def get_data_collator(tokenizer, is_multilabel: bool, max_length: int = 128):
    """Return the appropriate data collator."""
    if is_multilabel:
        return MultiLabelDataCollator(tokenizer=tokenizer, max_length=max_length)
    return DataCollatorWithPadding(tokenizer=tokenizer)


# ──────────────────────────────────────────────────────────────
# Weighted Trainer (BCEWithLogitsLoss + pos_weight)
# ──────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer that supports class weighting for imbalanced datasets."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight.clone().detach() if pos_weight is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits

        if self.pos_weight is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(model.device))
            loss = loss_fct(logits, labels.float())
        elif model.config.problem_type == "multi_label_classification":
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────
# Training arguments builder
# ──────────────────────────────────────────────────────────────

def build_training_args(
    output_dir: str,
    train_config: dict,
    *,
    overrides: dict | None = None,
    logging_dir: str | None = None,
    report_to: str = "tensorboard",
    load_best: bool = True,
) -> TrainingArguments:
    """
    Build a TrainingArguments instance from the YAML training config,
    with optional per-call overrides (e.g. from Optuna best params).
    """
    cfg = {**train_config}
    if overrides:
        cfg.update(overrides)

    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 16),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        weight_decay=cfg.get("weight_decay", 0.01),
        eval_strategy=cfg.get("evaluation_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=load_best,
        metric_for_best_model="f1_weighted",
        logging_dir=logging_dir or os.path.join(output_dir, "logs"),
        logging_steps=cfg.get("logging_steps", 100),
        report_to=report_to,
        seed=cfg.get("seed", 42),
    )


# ──────────────────────────────────────────────────────────────
# Evaluation + save helper
# ──────────────────────────────────────────────────────────────

def evaluate_and_save(trainer, test_dataset, tokenizer, output_dir, label_names=None,
                      is_multilabel=True, save_predictions=False):
    """Run prediction on the test set, save metrics and model."""
    print("\n--- Final Evaluation on Test Set ---")
    predictions_output = trainer.predict(test_dataset)
    print("\nTest Set Metrics:", predictions_output.metrics)

    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)

    # Save test metrics
    with open(os.path.join(final_model_path, "test_metrics.json"), "w") as f:
        json.dump(predictions_output.metrics, f, indent=4)
    print(f"✓ Test metrics saved to {final_model_path}/test_metrics.json")

    if save_predictions:
        np.save(os.path.join(output_dir, "test_predictions.npy"), predictions_output.predictions)
        np.save(os.path.join(output_dir, "test_true_labels.npy"), predictions_output.label_ids)

    # Save model + tokenizer
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✓ Model saved to {final_model_path}")

    return predictions_output


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# Common training pipeline (shared by baseline / clean / MLM)
# ──────────────────────────────────────────────────────────────

def run_optuna_search(
    trainer: Trainer,
    config: dict,
    output_dir: str,
) -> dict:
    """
    Run Optuna hyperparameter search if enabled in config.

    Returns the best hyperparameters dict (either from Optuna or defaults).
    """
    train_cfg = config["training"]

    if config.get("optuna", {}).get("enabled", False):
        print("\n--- Starting Hyperparameter Search with Optuna ---")
        best_trial = trainer.hyperparameter_search(
            direction=config["optuna"].get("direction", "maximize"),
            backend="optuna",
            hp_space=lambda trial: {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size", [16, 32]
                ),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
            },
            n_trials=config["optuna"].get("n_trials", 3),
        )
        print(f"\n✓ Best trial: {best_trial}")

        hp_path = os.path.join(output_dir, "best_hyperparameters.json")
        with open(hp_path, "w") as f:
            json.dump(best_trial.hyperparameters, f, indent=4)
        return best_trial.hyperparameters

    # Fallback: use config defaults
    return {
        "learning_rate": train_cfg["learning_rate"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "num_train_epochs": train_cfg["num_train_epochs"],
    }


def count_label_distribution(dataset_split, label_names, is_multilabel):
    """Compute label frequency counts for a dataset split."""
    from collections import Counter

    if is_multilabel:
        counts: dict[str, int] = Counter()
        for sample_labels in dataset_split["labels"]:
            for idx, val in enumerate(sample_labels):
                if val == 1:
                    counts[label_names[idx]] += 1
    else:
        counts = Counter(label_names[int(l)] for l in dataset_split["labels"])
    return counts


def run_standard_pipeline(
    config: dict,
    output_dir: str,
    *,
    model_name_or_path: str | None = None,
    extra_training_overrides: dict | None = None,
):
    """
    End-to-end standard training pipeline used by baseline, clean-data and
    MLM fine-tuning scripts.

    Steps: load data → tokenize → label distribution plot → Optuna search
    → final training → evaluation → save model.

    Args:
        config: Full YAML config dict.
        output_dir: Where to save results.
        model_name_or_path: Override the model checkpoint to fine-tune
                            (e.g. MLM-adapted model path).
        extra_training_overrides: Extra TrainingArguments overrides.

    Returns:
        (trainer, predictions_output, label_info)
    """
    from src.data_utils import load_and_preprocess_dataset, tokenize_dataset
    from src.model_utils import get_model
    from src.metrics import make_compute_metrics
    from src.viz_utils import visualize_embeddings, plot_label_distribution

    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    base_model = model_name_or_path or config["model"]["name"]
    max_length = config["model"].get("max_length", 128)
    seed = train_cfg.get("seed", 42)

    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    # 1. Load & preprocess
    dataset, label_info = load_and_preprocess_dataset(dataset_cfg)
    label_names = label_info["label_names"]
    num_labels = label_info["num_labels"]
    is_multilabel = label_info["is_multilabel"]

    # 2. Tokenize
    tokenized_datasets, tokenizer = tokenize_dataset(
        dataset,
        text_column=dataset_cfg["text_column"],
        model_name=config["model"]["name"],
        max_length=max_length,
    )

    # 3. Label distribution plot
    label_counts = count_label_distribution(dataset["train"], label_names, is_multilabel)
    plot_label_distribution(label_counts, output_dir)

    # 4. Model init factory
    def model_init(trial=None):
        return get_model(
            model_name=base_model,
            num_labels=num_labels,
            is_multilabel=is_multilabel,
            label_names=label_names,
        )

    # 5. Data collator & metrics
    data_collator = get_data_collator(tokenizer, is_multilabel, max_length)
    compute_metrics_fn = make_compute_metrics(is_multilabel)

    # 6. Search-phase trainer (for Optuna)
    search_args = build_training_args(
        output_dir=os.path.join(output_dir, "results"),
        train_config=train_cfg,
        logging_dir=os.path.join(output_dir, "logs"),
    )

    search_trainer = Trainer(
        model_init=model_init,
        args=search_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    best_params = run_optuna_search(search_trainer, config, output_dir)

    # 7. Final training with best params
    print("\n--- Final Training ---")
    merged_overrides = {**best_params, **(extra_training_overrides or {})}

    final_args = build_training_args(
        output_dir=os.path.join(output_dir, "final_results"),
        train_config=train_cfg,
        overrides=merged_overrides,
        logging_dir=os.path.join(output_dir, "final_logs"),
    )

    final_trainer = Trainer(
        model_init=model_init,
        args=final_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    final_trainer.train()

    # 8. Evaluate & save
    predictions_output = evaluate_and_save(
        final_trainer,
        tokenized_datasets["test"],
        tokenizer,
        output_dir,
        label_names=label_names,
        is_multilabel=is_multilabel,
    )

    # 9. Embedding visualisation
    visualize_embeddings(
        final_trainer.model,
        tokenized_datasets["test"],
        tokenizer,
        label_names,
        output_dir,
    )

    return final_trainer, predictions_output, label_info
