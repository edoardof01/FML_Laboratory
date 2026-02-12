#model_utils.py
import torch
from transformers import AutoModelForSequenceClassification

def get_model(model_name, num_labels, is_multilabel, peft_config=None, label_names=None):
    """
    Initialize a model for sequence classification.
    """
    if is_multilabel:
        problem_type = "multi_label_classification"
    else:
        problem_type = "single_label_classification"
        
    print(f"Loading Model: {model_name}")
    print(f"  Problem Type: {problem_type}")
    print(f"  Num Labels: {num_labels}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type
    )
    
    # Configure label mapping
    if label_names:
        model.config.id2label = {i: label for i, label in enumerate(label_names)}
        model.config.label2id = {label: i for i, label in enumerate(label_names)}
    
    # Print trainable parameters (all are trainable now)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (100%)")
        
    return model

