"""Evaluation metrics for stroke classification."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    """Run full evaluation on a dataloader.

    Returns:
        Dict with macro_f1, per_class_metrics, confusion_matrix, all_preds, all_labels
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy = (all_preds == all_labels).mean()

    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)

    target_names = class_names or [str(i) for i in range(len(per_class_f1))]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0
    )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm,
        "classification_report": report,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def print_results(results: dict, class_names: list[str] | None = None):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"Weighted-F1: {results['weighted_f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(results["classification_report"])
