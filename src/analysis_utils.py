# analysis_utils.py
"""Embedding analysis and hard-negative mining utilities."""
import torch
import numpy as np
from tqdm import tqdm


def get_embeddings(model, dataset, tokenizer, batch_size=32, device="cuda"):
    """
    Compute [CLS] embeddings for a dataset.

    Returns:
        (embeddings, labels) as numpy arrays.
    """
    model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []

    print(f"--- Computing Embeddings ({len(dataset)} samples) ---")

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        texts = batch.get("text") or batch.get("sentence")
        if "labels" in batch:
            all_labels.extend(batch["labels"])

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings), np.array(all_labels)


def find_hard_negatives(model, dataset, tokenizer, top_k=5, device="cuda"):
    """Identify samples where the model is confident but wrong."""
    model.to(device)
    model.eval()

    hard_negatives = []
    print("--- Mining Hard Negatives ---")

    for sample in tqdm(dataset):
        text = sample["text"]
        true_labels = sample["labels"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        if model.config.problem_type == "multi_label_classification":
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            false_positives = np.where((preds == 1) & (np.array(true_labels) == 0))[0]
            for fp_idx in false_positives:
                confidence = probs[fp_idx]
                if confidence > 0.8:
                    hard_negatives.append({
                        "text": text,
                        "true_labels": true_labels,
                        "predicted_label": int(fp_idx),
                        "confidence": float(confidence),
                        "type": "False Positive",
                    })
        else:
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_id = int(np.argmax(probs))
            true_id = true_labels

            if pred_id != true_id and probs[pred_id] > 0.8:
                hard_negatives.append({
                    "text": text,
                    "true_label": true_id,
                    "predicted_label": pred_id,
                    "confidence": float(probs[pred_id]),
                    "type": "Misclassification",
                })

    hard_negatives.sort(key=lambda x: x["confidence"], reverse=True)
    return hard_negatives[:top_k]
