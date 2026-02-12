#analysis_utils.py
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

def get_embeddings(model, dataset, tokenizer, batch_size=32, device="cuda"):
    """
    Compute embeddings for a dataset using the [CLS] token of the model.
    """
    model.to(device)
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    print(f"--- Computing Embeddings ({len(dataset)} samples) ---")
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        texts = batch["text"] if "text" in batch else batch["sentence"]
        if "labels" in batch:
            labels = batch["labels"]
            all_labels.extend(labels)
            
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use the hidden state of the [CLS] token from the last layer
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            
    return np.vstack(all_embeddings), np.array(all_labels)

def find_hard_negatives(model, dataset, tokenizer, top_k=5, device="cuda"):
    """
    Identify samples where the model is confident but wrong.
    """
    model.to(device)
    model.eval()
    
    hard_negatives = []
    
    print("--- Mining Hard Negatives ---")
    
    # Process one by one or small batches for detailed analysis
    for idx, sample in enumerate(tqdm(dataset)):
        text = sample["text"]
        true_labels = sample["labels"] # Multi-hot vector or class ID
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Multi-label logic
        if model.config.problem_type == "multi_label_classification":
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            # Let's focus on False Positives with high confidence
            false_positives = np.where((preds == 1) & (np.array(true_labels) == 0))[0]
            
            for fp_idx in false_positives:
                confidence = probs[fp_idx]
                if confidence > 0.8: # Threshold for "confident"
                    hard_negatives.append({
                        "text": text,
                        "true_labels": true_labels,
                        "predicted_label": fp_idx,
                        "confidence": float(confidence),
                        "type": "False Positive"
                    })
                    
        else:
            # Single-label logic
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_id = np.argmax(probs)
            true_id = true_labels
            
            if pred_id != true_id:
                confidence = probs[pred_id]
                if confidence > 0.8:
                    hard_negatives.append({
                        "text": text,
                        "true_label": true_id,
                        "predicted_label": pred_id,
                        "confidence": float(confidence),
                        "type": "Misclassification"
                    })
                    
    # Sort by confidence
    hard_negatives.sort(key=lambda x: x["confidence"], reverse=True)
    return hard_negatives[:top_k]
