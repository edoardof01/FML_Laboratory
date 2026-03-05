# viz_utils.py
"""Visualisation utilities: label distributions, confusion matrices, embedding plots."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def plot_label_distribution(label_counts, output_dir, title="Label Distribution"):
    """Plot bar chart of top-20 label distribution."""
    plt.figure(figsize=(14, 6))
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    if not sorted_items:
        return

    labels, counts = zip(*sorted_items)
    df = pd.DataFrame({"Label": labels, "Count": counts})

    sns.barplot(data=df, x="Label", y="Count", palette="viridis")
    plt.title(f"{title} (Top 20)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_distribution.png"))
    plt.close()
    print(f"✓ Distribution plot saved to {output_dir}")


def plot_confusion_matrix(true_labels, predicted_labels, label_names, output_dir):
    """Plot normalised confusion matrix."""
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    if len(label_names) <= 20:
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized, display_labels=label_names
        )
        disp.plot(cmap="Blues", xticks_rotation="vertical", values_format=".2f")
        plt.title("Normalized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"))
        plt.close()
        print("✓ Confusion matrix saved")


def get_cls_embeddings(model, dataset, tokenizer, sample_size=2000, batch_size=32):
    """
    Extract [CLS] embeddings from the last hidden state.

    Shared helper used by both visualise_embeddings() and analysis_utils.

    Returns:
        (embeddings, labels) as numpy arrays.
    """
    if len(dataset) > sample_size:
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        dataset = dataset.select(indices)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    model.eval()
    device = next(model.parameters()).device

    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, output_hidden_states=True)
            cls_emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)
            all_labels.append(labels.numpy())

    return np.vstack(all_embeddings), np.vstack(all_labels)


def visualize_embeddings(model, dataset, tokenizer, label_names, output_dir, sample_size=2000):
    """Generate and plot PCA and t-SNE of model embeddings."""
    print("\n--- Embedding Visualization ---")

    embeddings, labels = get_cls_embeddings(model, dataset, tokenizer, sample_size)

    # Determine primary label for colouring
    primary_labels = []
    for lbl in labels:
        if lbl.size == 1:
            primary_labels.append(label_names[int(lbl)])
        else:
            idxs = np.where(lbl == 1)[0]
            primary_labels.append(label_names[idxs[0]] if len(idxs) > 0 else "None")

    # PCA
    print("Running PCA...")
    emb_pca = PCA(n_components=2).fit_transform(embeddings)
    _plot_scatter(emb_pca, primary_labels, "PCA", output_dir)

    # t-SNE
    print("Running t-SNE...")
    emb_tsne = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)
    _plot_scatter(emb_tsne, primary_labels, "t-SNE", output_dir)


def _plot_scatter(points, labels, method, output_dir):
    df = pd.DataFrame(points, columns=["Dim1", "Dim2"])
    df["Label"] = labels

    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Label", palette="viridis", alpha=0.7, s=20)
    plt.title(f"{method} Visualization")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"embeddings_{method.lower()}.png"))
    plt.close()
    print(f"✓ {method} plot saved")
