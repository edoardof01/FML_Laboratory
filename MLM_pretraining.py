# MLM_pretraining.py
import os
import json
import torch
import optuna
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification, 
    DataCollatorForLanguageModeling,
    TrainingArguments, 
    Trainer, 
    IntervalStrategy
)
from datasets import Dataset

# Use our new modules
from src.data_utils import load_and_preprocess_dataset, tokenize_dataset
from src.metrics import compute_metrics
from src.train_utils import MultiLabelDataCollator, set_seed
from src.config_utils import get_config

# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================
config = get_config()
DATASET_CONFIG = config["dataset"]
OUTPUT_DIR = config["output_dir"].replace("go-emotions", "continued_pretraining_go-emotions") # Adjusted for MLM
TRAIN_CONFIG = config["training"]
MODEL_CONFIG = config["model"]

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(TRAIN_CONFIG.get("seed", 42))

print(f"\n{'='*60}")
print(f"DATASET: {DATASET_CONFIG['name']}")
print(f"OUTPUT: {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ============================================================
# 2. LOAD & PREPROCESS
# ============================================================
dataset, label_info = load_and_preprocess_dataset(DATASET_CONFIG)
label_names = label_info["label_names"]
num_labels = label_info["num_labels"]
is_multilabel = label_info["is_multilabel"]

# ============================================================
# 3. CONTINUED PRETRAINING (MLM)
# ============================================================
print("\n--- Phase 1: Continued Pretraining with Masked Language Modeling ---")

mlm_adapted_model_path = os.path.join(OUTPUT_DIR, "mlm_adapted_model")

# Setup tokenizer once
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["name"])

if not os.path.exists(mlm_adapted_model_path):
    print("Domain-adapted model not found. Starting pretraining...")

    # Create unlabeled corpus from all splits
    text_column = DATASET_CONFIG["text_column"]
    corpus = []
    for split in dataset.values():
        corpus.extend(split[text_column])
    
    print(f"Corpus size: {len(corpus)} documents")
    unlabeled_dataset = Dataset.from_dict({text_column: corpus})

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column], 
            truncation=True, 
            max_length=MODEL_CONFIG.get("max_length", 128), 
            return_special_tokens_mask=True
        )

    tokenized_unlabeled_dataset = unlabeled_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=[text_column]
    )
    
    data_collator_mlm = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm_probability=0.15
    )

    mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_CONFIG["name"])
    
    training_args_mlm = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "mlm_results"),
        overwrite_output_dir=True,
        per_device_train_batch_size=64,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=100,
        report_to="tensorboard",
    )

    trainer_mlm = Trainer(
        model=mlm_model,
        args=training_args_mlm,
        data_collator=data_collator_mlm,
        train_dataset=tokenized_unlabeled_dataset,
    )
    
    trainer_mlm.train()
    
    # Save domain-adapted model
    os.makedirs(mlm_adapted_model_path, exist_ok=True)
    trainer_mlm.save_model(mlm_adapted_model_path)
    tokenizer.save_pretrained(mlm_adapted_model_path)
    print(f"\n✓ Domain-adapted DistilBERT model saved to: {mlm_adapted_model_path}")

else:
    print(f"\n✓ Domain-adapted model found at '{mlm_adapted_model_path}'. Skipping pretraining.")


# ============================================================
# 4. SUPERVISED FINE-TUNING
# ============================================================
print("\n--- Phase 2: Supervised Fine-Tuning for Classification ---")

# Tokenize for classification
tokenized_datasets, _ = tokenize_dataset(
    dataset, 
    text_column=DATASET_CONFIG["text_column"], 
    model_name=MODEL_CONFIG["name"],
    max_length=MODEL_CONFIG.get("max_length", 128)
)

# Data Collator
if is_multilabel:
    data_collator = MultiLabelDataCollator(tokenizer=tokenizer, max_length=MODEL_CONFIG.get("max_length", 128))
else:
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# wrapper for metrics
def compute_metrics_wrapper(eval_pred):
    return compute_metrics(eval_pred, is_multilabel=is_multilabel)

# Load MLM-adapted model function
if is_multilabel:
    problem_type = "multi_label_classification"
else:
    problem_type = "single_label_classification"

def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(
        mlm_adapted_model_path,
        num_labels=num_labels,
        problem_type=problem_type
    )
    model.config.id2label = {i: name for i, name in enumerate(label_names)}
    model.config.label2id = {name: i for i, name in enumerate(label_names)}
    return model

# ============================================================
# 4.1. HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================
if config.get("optuna", {}).get("enabled", False):
    print("\n--- Starting Hyperparameter Search with Optuna ---")
    
    training_args_hp = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "hp_search"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=100,
        report_to="none",
        seed=TRAIN_CONFIG.get("seed", 42)
    )
    
    trainer_hp = Trainer(
        model_init=model_init,
        args=training_args_hp,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
    )
    
    best_trial = trainer_hp.hyperparameter_search(
        direction=config["optuna"].get("direction", "maximize"),
        backend="optuna",
        hp_space=lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
        },
        n_trials=config["optuna"].get("n_trials", 3)
    )
    print(f"\n✓ Best trial: {best_trial}")
    
    # Save best parameters
    with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.json"), 'w') as f:
        json.dump(best_trial.hyperparameters, f, indent=4)
        
    final_params = best_trial.hyperparameters
else:
    final_params = {
        "learning_rate": TRAIN_CONFIG["learning_rate"],
        "per_device_train_batch_size": TRAIN_CONFIG["per_device_train_batch_size"],
        "num_train_epochs": 3  # Fixed to 3 for faster training
    }
    print("\n--- Using default hyperparameters (no Optuna) ---")
    print(f"  Learning rate: {final_params['learning_rate']}")
    print(f"  Batch size: {final_params['per_device_train_batch_size']}")
    print(f"  Epochs: {final_params['num_train_epochs']}")

# ============================================================
# 4.2. FINAL FINE-TUNING
# ============================================================
print("\n--- Final Fine-tuning with Best Hyperparameters ---")

training_args_finetune = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "finetuning_results"),
    learning_rate=final_params.get("learning_rate", TRAIN_CONFIG["learning_rate"]),
    per_device_train_batch_size=final_params.get("per_device_train_batch_size", TRAIN_CONFIG["per_device_train_batch_size"]),
    per_device_eval_batch_size=final_params.get("per_device_train_batch_size", TRAIN_CONFIG["per_device_eval_batch_size"]),
    num_train_epochs=final_params.get("num_train_epochs", 3),
    weight_decay=TRAIN_CONFIG["weight_decay"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_dir=os.path.join(OUTPUT_DIR, "finetuning_logs"),
    logging_steps=100,
    report_to="tensorboard",
    seed=TRAIN_CONFIG.get("seed", 42)
)

trainer_finetune = Trainer(
    model_init=model_init,
    args=training_args_finetune,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_wrapper,
)

trainer_finetune.train()
print("✓ Fine-tuning complete")

# ============================================================
# 5. EVALUATION
# ============================================================
print("\n--- Final Evaluation on Test Set ---")
predictions_output = trainer_finetune.predict(tokenized_datasets["test"])
print("\nTest Set Metrics:", predictions_output.metrics)

# Save test metrics to JSON
test_metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
with open(test_metrics_path, 'w') as f:
    json.dump(predictions_output.metrics, f, indent=4)
print(f"✓ Test metrics saved to: {test_metrics_path}")

# Save Final
final_model_dir = os.path.join(OUTPUT_DIR, "final_distilbert_model_with_mlm")
os.makedirs(final_model_dir, exist_ok=True)
trainer_finetune.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"\n✓ Final model saved to: {final_model_dir}")

# Save Config
config_dict = {
    "dataset": DATASET_CONFIG["name"],
    "is_multilabel": is_multilabel,
    "num_labels": num_labels,
    "label_names": label_names
}
with open(os.path.join(final_model_dir, "config.json"), 'w') as f:
    json.dump(config_dict, f, indent=4)

