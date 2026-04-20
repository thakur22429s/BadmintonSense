"""Visualization utilities for training curves, confusion matrices, and embeddings."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """Plot and save a confusion matrix heatmap."""
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history: dict,
    save_path: Path,
    title: str = "Training Curves",
):
    """Plot training/validation loss and F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1 curve
    ax2.plot(epochs, history["val_f1"], label="Val Macro-F1", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(
    f1_scores: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "Per-class F1 Score",
):
    """Bar chart of per-class F1 scores."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(range(len(class_names)), f1_scores, color=colors)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{score:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    save_path: Path,
    perplexity: int = 30,
    title: str = "t-SNE of Learned Embeddings",
):
    """t-SNE visualization of model embeddings colored by class."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        name = class_names[label] if label < len(class_names) else str(label)
        ax.scatter(coords[mask, 0], coords[mask, 1], label=name, alpha=0.6, s=20)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
