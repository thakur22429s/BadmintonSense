"""Loss functions for stroke classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing and optional per-class weights."""

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        # Buffer so it follows .to(device)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)

        if self.class_weights is not None:
            nll_loss = F.nll_loss(log_probs, targets, weight=self.class_weights, reduction="none")
            # Weight smooth loss too — average log prob of all classes weighted
            smooth_loss = -(log_probs * self.class_weights).mean(dim=-1)
        else:
            nll_loss = F.nll_loss(log_probs, targets, reduction="none")
            smooth_loss = -log_probs.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def compute_class_weights(labels, num_classes: int, scheme: str = "inverse_sqrt") -> torch.Tensor:
    """Compute per-class weights for imbalanced datasets.

    Args:
        labels: array-like of int labels (numpy or list).
        num_classes: total class count.
        scheme: "inverse" (1/freq) or "inverse_sqrt" (1/sqrt(freq)) — sqrt is gentler.

    Returns:
        Tensor of shape (num_classes,) normalized to mean=1.
    """
    import numpy as np

    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for empty classes

    if scheme == "inverse":
        weights = 1.0 / counts
    else:  # inverse_sqrt
        weights = 1.0 / np.sqrt(counts)

    # Normalize so mean weight = 1 (preserves loss scale)
    weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)
