# contrastive_pretraining.py
import os
import torch
import json
import optuna
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from src.data_utils import load_and_preprocess_dataset, tokenize_dataset
from src.config_utils import get_config
from src.losses import SupConLoss
from src.train_utils import set_seed, MultiLabelDataCollator
from tqdm import tqdm

# ============================================================
# 1. SETUP
# ============================================================
config = get_config()
DATASET_CONFIG = config["dataset"]
OUTPUT_DIR = os.path.join(config["output_dir"], "supcon_pretraining")
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(42)

print(f"\n{'='*60}")
print(f"SUPERVISED CONTRASTIVE PRETRAINING (SupCon)")
print(f"{'='*60}\n")

# ============================================================
# 2. DATASET
# ============================================================
dataset, label_info = load_and_preprocess_dataset(DATASET_CONFIG)
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

# We need a custom collator that returns JUST inputs and labels (no formatting for Trainer)
def collate_fn(batch):
    texts = [item[DATASET_CONFIG["text_column"]] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])
    
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return inputs, labels

train_dataset = dataset["train"]
train_loader = DataLoader(
    train_dataset, 
    batch_size=config["training"]["per_device_train_batch_size"], 
    shuffle=True, 
    collate_fn=collate_fn
)

# ============================================================
# 3. MODEL (ENCODER ONLY)
# ============================================================
# For SupCon, we use the base model (without classification head) + a Projection Head
class SupConModel(torch.nn.Module):
    def __init__(self, model_name, dim=768, feat_dim=128):
        super(SupConModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim, feat_dim)
        )

    def forward(self, input_ids, attention_mask, return_dual_views=False):
        if return_dual_views:
            # Generate two views using dropout stochasticity
            self.train()  # Enable dropout
            outputs1 = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            outputs2 = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            feat1 = outputs1.last_hidden_state[:, 0, :]
            feat2 = outputs2.last_hidden_state[:, 0, :]
            feat1 = torch.nn.functional.normalize(feat1, dim=1)
            feat2 = torch.nn.functional.normalize(feat2, dim=1)
            
            proj1 = self.head(feat1)
            proj2 = self.head(feat2)
            proj1 = torch.nn.functional.normalize(proj1, dim=1)
            proj2 = torch.nn.functional.normalize(proj2, dim=1)
            
            # Stack as [bsz, 2, dim]
            return torch.stack([proj1, proj2], dim=1)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            feat = outputs.last_hidden_state[:, 0, :] # [CLS] token
            feat = torch.nn.functional.normalize(feat, dim=1) # Normalize embeddings
            projections = self.head(feat)
            projections = torch.nn.functional.normalize(projections, dim=1)
            return projections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 4. HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================
def train_supcon(temperature, learning_rate, epochs=3):
    """Train SupCon model with given hyperparameters and return validation loss"""
    model = SupConModel(config["model"]["name"]).to(device)
    criterion = SupConLoss(temperature=temperature, use_jaccard=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, labels in train_loader:
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.to(device)
            
            # Forward with dual-view augmentation (dropout-based)
            features = model(input_ids, attention_mask, return_dual_views=True)  # [bsz, 2, dim]
            
            # Loss with Jaccard-based similarity
            loss = criterion(features, labels=labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
    
    avg_loss = epoch_loss / (len(train_loader) * epochs)
    return model, avg_loss

if config.get("optuna", {}).get("enabled", False):
    print("\n--- Hyperparameter Optimization with Optuna ---")
    
    def optuna_objective(trial):
        temperature = trial.suggest_float("temperature", 0.05, 0.15)
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True)
        
        _, avg_loss = train_supcon(temperature, learning_rate, epochs=3)
        return avg_loss  # Minimize loss
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(optuna_objective, n_trials=config.get("optuna", {}).get("n_trials", 2))
    
    print(f"\n✓ Best hyperparameters: {study.best_trial.params}")
    best_params = study.best_trial.params
    
    with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=4)
else:
    # Use improved defaults
    best_params = {
        "temperature": 0.07,
        "learning_rate": 1e-4
    }
    print("\n--- Using default hyperparameters (no Optuna) ---")
    print(f"  Temperature: {best_params['temperature']}")
    print(f"  Learning Rate: {best_params['learning_rate']}")

# ============================================================
# 5. FINAL SUPCON TRAINING
# ============================================================
print("\n--- Final SupCon Training with Best Hyperparameters ---")
epochs = 3  # Keeping low for time constraints
model, final_loss = train_supcon(
    temperature=best_params["temperature"],
    learning_rate=best_params["learning_rate"],
    epochs=epochs
)
print(f"✓ Final training loss: {final_loss:.4f}")

# ============================================================
# 6. SAVE ENCODER
# ============================================================
save_path = os.path.join(OUTPUT_DIR, "supcon_encoder")
os.makedirs(save_path, exist_ok=True)
model.encoder.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n✓ Saved SupCon encoder to {save_path}")

# ============================================================
# 7. FINE-TUNING FOR CLASSIFICATION
# ============================================================
print("\n--- Phase 2: Fine-tuning SupCon encoder for classification ---")

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from src.metrics import compute_metrics
from src.train_utils import MultiLabelDataCollator
import json

# Tokenize datasets for classification
from src.data_utils import tokenize_dataset
tokenized_datasets, _ = tokenize_dataset(
    dataset,
    text_column=DATASET_CONFIG["text_column"],
    model_name=config["model"]["name"],
    max_length=config["model"].get("max_length", 128)
)

# Setup
num_labels = label_info["num_labels"]
is_multilabel = label_info["is_multilabel"]
label_names = label_info["label_names"]

if is_multilabel:
    problem_type = "multi_label_classification"
    data_collator = MultiLabelDataCollator(tokenizer=tokenizer, max_length=config["model"].get("max_length", 128))
else:
    problem_type = "single_label_classification"
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics_wrapper(eval_pred):
    return compute_metrics(eval_pred, is_multilabel=is_multilabel)

# Load SupCon pretrained encoder for classification
model_for_finetune = AutoModelForSequenceClassification.from_pretrained(
    save_path,
    num_labels=num_labels,
    problem_type=problem_type
)
model_for_finetune.config.id2label = {i: name for i, name in enumerate(label_names)}
model_for_finetune.config.label2id = {name: i for i, name in enumerate(label_names)}

training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "finetuning_results"),
    learning_rate=float(config["training"]["learning_rate"]),
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"],
    weight_decay=config["training"]["weight_decay"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=100,
    report_to="tensorboard",
    seed=config["training"].get("seed", 42)
)

trainer = Trainer(
    model=model_for_finetune,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_wrapper,
)

trainer.train()
print("✓ Fine-tuning complete")

# ============================================================
# 8. EVALUATION & SAVE
# ============================================================
print("\n--- Final Evaluation on Test Set ---")
predictions_output = trainer.predict(tokenized_datasets["test"])
print("\nTest Set Metrics:", predictions_output.metrics)

# Save test metrics
test_metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
with open(test_metrics_path, 'w') as f:
    json.dump(predictions_output.metrics, f, indent=4)
print(f"✓ Test metrics saved to: {test_metrics_path}")

# Save final model
final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
os.makedirs(final_model_dir, exist_ok=True)
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"✓ Final SupCon-pretrained model saved to: {final_model_dir}")

