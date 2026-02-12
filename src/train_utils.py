#train_utils.py
import torch
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Trainer

@dataclass
class MultiLabelDataCollator:
    """
    Data collator that handles multi-label classification properly.
    """
    tokenizer: Any
    padding: Union[bool, str] = True
    max_length: int = 128
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate labels from features
        labels = [feature.pop("labels") for feature in features] if "labels" in features[0] else None
        
        # Pad text features
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Add labels back (properly formatted)
        if labels is not None:
             if isinstance(labels[0], torch.Tensor):
                 if labels[0].dim() == 0:
                     batch["labels"] = torch.stack(labels)
                 else:
                     batch["labels"] = torch.stack(labels).float()
             elif isinstance(labels[0], list):
                 # Assume float for multi-label binary vectors
                 batch["labels"] = torch.tensor(labels, dtype=torch.float)
             else:
                 batch["labels"] = torch.tensor(labels, dtype=torch.long)
                 
        return batch

class WeightedTrainer(Trainer):
    """
    Trainer that supports class weighting for imbalanced datasets.
    """
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight.clone().detach() if pos_weight is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.pos_weight is not None:
            # Weighted BCEWithLogitsLoss
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(model.device))
            # Ensure labels are float for BCE
            labels = labels.float()
            loss = loss_fct(logits, labels)
        else:
            # Standard loss (CrossEntropy or BCE based on model config)
            if model.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                labels = labels.float()
                loss = loss_fct(logits, labels)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                
        return (loss, outputs) if return_outputs else loss

def set_seed(seed=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
