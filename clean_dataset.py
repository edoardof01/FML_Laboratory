# clean_dataset.py
import os
import torch
import numpy as np
import cleanlab
from cleanlab.filter import find_label_issues
from cleanlab.dataset import health_summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.data_utils import load_and_preprocess_dataset, tokenize_dataset
from src.config_utils import get_config
from src.model_utils import get_model
from src.train_utils import MultiLabelDataCollator

# ============================================================
# 1. SETUP
# ============================================================
config = get_config()
DATASET_CONFIG = config["dataset"]
OUTPUT_DIR = os.path.join(config["output_dir"], "cleanlab_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"CONFIDENT LEARNING (DATASET CLEANING)")
print(f"Dataset: {DATASET_CONFIG['name']}")
print(f"Output: {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ============================================================
# 2. LOAD DATASET
# ============================================================
dataset, label_info = load_and_preprocess_dataset(DATASET_CONFIG)
label_names = label_info["label_names"]
num_labels = label_info["num_labels"]
is_multilabel = label_info["is_multilabel"]

# Tokenize
tokenized_datasets, tokenizer = tokenize_dataset(
    dataset, 
    text_column=DATASET_CONFIG["text_column"], 
    model_name=config["model"]["name"]
)
tokenized_train = tokenized_datasets["train"]

# ============================================================
# 3. COMPUTE OUT-OF-SAMPLE PROBABILITIES (CROSS-VALIDATION)
# ============================================================
print("\n--- Computing Out-of-Sample Probabilities via Cross-Validation ---")
# Ideally we use K-Fold, but for speed in this demo we might just use a hold-out or a simple split.
# To do it properly with CleanLab, we usually need cross-val predictions.
# Here we simulate CV by splitting train into K folds manually or using a library.

from sklearn.model_selection import KFold
from torch.utils.data import Subset

n_splits = 3  # Valid value for demo
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

all_probs = np.zeros((len(tokenized_train), num_labels))
indices = np.arange(len(tokenized_train))

# Data Collator
if is_multilabel:
    data_collator = MultiLabelDataCollator(tokenizer=tokenizer)
else:
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n[Fold {fold+1}/{n_splits}]")
    
    # Subset
    train_sub = tokenized_train.select(train_idx)
    val_sub = tokenized_train.select(val_idx)
    
    # Init Model (Fresh each fold)
    model = get_model(
        model_name=config["model"]["name"],
        num_labels=num_labels,
        is_multilabel=is_multilabel,
        peft_config=config.get("peft"), # Use PEFT for speed if enabled
        label_names=label_names
    )
    
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"fold_{fold}"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1, # Reduced for demo speed
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        disable_tqdm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_sub,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Predict on validation chunk
    print("  Predicting on validation fold...")
    preds_output = trainer.predict(val_sub)
    logits = preds_output.predictions
    
    if is_multilabel:
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
    else:
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
    all_probs[val_idx] = probs

#Save probabilities
np.save(os.path.join(OUTPUT_DIR, "train_probs.npy"), all_probs)

# ============================================================
# 4. FIND LABEL ISSUES WITH CLEANLAB
# ============================================================
print("\n--- Identifying Label Issues ---")

if is_multilabel:
    # For multi-label, cleanlab expects a list of lists containing indices of active labels
    # Convert binary vectors to list of indices: [1,0,1,0] -> [0, 2]
    binary_labels = np.array(tokenized_train["labels"])
    labels = [list(np.where(row == 1)[0]) for row in binary_labels]
    label_issues = find_label_issues(
        labels=labels,
        pred_probs=all_probs,
        multi_label=True
    )
else:
    labels = np.array(tokenized_train["labels"])
    label_issues = find_label_issues(
        labels=labels,
        pred_probs=all_probs
    )

print(f"\nFound {label_issues.sum()} potential label issues.")
print(f"Saving issues to {os.path.join(OUTPUT_DIR, 'label_issues.csv')}")

# Get indices of problematic samples (label_issues is a boolean mask for multi-label)
issue_indices = np.where(label_issues)[0]

# Get raw dataset for text column
raw_train = dataset[DATASET_CONFIG["split_names"]["train"]]

# Add text for context
import pandas as pd
issues_df = pd.DataFrame({
    "index": issue_indices,
    "text": [raw_train[int(i)][DATASET_CONFIG["text_column"]] for i in issue_indices]
})

issues_df.to_csv(os.path.join(OUTPUT_DIR, "label_issues.csv"), index=False)

# ============================================================
# 5. SAVE CLEANED DATASET
# ============================================================
print("\n--- Saving Cleaned Dataset ---")
indices_to_remove = set(issue_indices)
all_indices = set(range(len(tokenized_train)))
clean_indices = sorted(list(all_indices - indices_to_remove))

print(f"Original train size: {len(tokenized_train)}")
print(f"Cleaned train size: {len(clean_indices)}")

# Filter the training split
cleaned_train = tokenized_train.select(clean_indices)

raw_train = dataset[DATASET_CONFIG["split_names"]["train"]]
cleaned_raw_train = raw_train.select(clean_indices)
dataset[DATASET_CONFIG["split_names"]["train"]] = cleaned_raw_train

# Save to disk
clean_dataset_path = os.path.join("data", f"{DATASET_CONFIG['name']}_cleaned")
os.makedirs(os.path.dirname(clean_dataset_path), exist_ok=True)
dataset.save_to_disk(clean_dataset_path)

print(f"\n✓ Cleaned dataset saved to: {clean_dataset_path}")
print(f"To use this dataset, add 'path: {clean_dataset_path}' to your config under 'dataset'.")

print("\n✓ Analysis and Cleaning Complete.")
