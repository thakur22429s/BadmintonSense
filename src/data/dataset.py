"""PyTorch Dataset and DataLoader with LOSO cross-validation splits."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class StrokeDataset(Dataset):
    """PyTorch Dataset for stroke pose sequences.

    Each sample is a flattened keypoint sequence: (T, K*3) tensor + integer label.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        config: dict | None = None,
    ):
        """
        Args:
            sequences: shape (N, T, K, 3)
            labels: shape (N,) integer class indices
            augment: Whether to apply online augmentation.
            config: Config dict (needed if augment=True).
        """
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.config = config

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq = self.sequences[idx].copy()  # (T, K, 3)

        if self.augment and self.config:
            from src.data.preprocessing import preprocess_single
            # Re-run augmentation on already-normalized data
            aug_config = self.config["data"]["augmentation"]
            if aug_config["enabled"]:
                seq = self._apply_augmentation(seq)

        # Flatten keypoints: (T, K, 3) -> (T, K*3)
        T, K, D = seq.shape
        flat = seq.reshape(T, K * D)

        return torch.tensor(flat, dtype=torch.float32), int(self.labels[idx])

    def _apply_augmentation(self, seq: np.ndarray) -> np.ndarray:
        from src.data.preprocessing import (
            augment_temporal_jitter,
            augment_gaussian_noise,
            augment_horizontal_flip,
            augment_temporal_scale,
            augment_frame_dropout,
        )

        aug_config = self.config["data"]["augmentation"]
        selected_indices = self.config["data"]["selected_keypoints"]

        if np.random.random() < 0.5:
            seq = augment_temporal_jitter(seq, aug_config["temporal_jitter_frames"])
        seq = augment_gaussian_noise(seq, aug_config["gaussian_noise_std"])
        if np.random.random() < aug_config["horizontal_flip_prob"]:
            seq = augment_horizontal_flip(seq, selected_indices)
        if np.random.random() < 0.3:
            seq = augment_temporal_scale(seq, tuple(aug_config["temporal_scale_range"]))
        if np.random.random() < 0.3:
            seq = augment_frame_dropout(seq, aug_config["frame_dropout_rate"])

        return seq


def get_loso_splits(
    metadata: pd.DataFrame,
    sequences: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    """Generate Leave-One-Session-Out cross-validation folds.

    Each fold uses one match as validation, rest as training.

    Returns:
        List of dicts, each with keys: train_indices, val_indices, fold_match_id
    """
    match_ids = metadata["match_id"].unique()
    folds = []

    for match_id in sorted(match_ids):
        val_mask = metadata["match_id"] == match_id
        val_indices = np.where(val_mask)[0]
        train_indices = np.where(~val_mask)[0]

        if len(val_indices) < 5:
            continue

        folds.append({
            "train_indices": train_indices,
            "val_indices": val_indices,
            "fold_match_id": match_id,
        })

    return folds


def get_stratified_split(
    metadata: pd.DataFrame,
    sequences: np.ndarray,
    labels: np.ndarray,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    """Standard stratified train/val/test split, respecting match boundaries.

    No match appears in multiple splits.

    Returns:
        Dict with keys: train_indices, val_indices, test_indices
    """
    from sklearn.model_selection import GroupShuffleSplit

    match_ids = metadata["match_id"].values
    unique_matches = metadata["match_id"].unique()

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_matches)

    n_matches = len(unique_matches)
    n_train = int(n_matches * ratios[0])
    n_val = int(n_matches * ratios[1])

    train_matches = set(unique_matches[:n_train])
    val_matches = set(unique_matches[n_train:n_train + n_val])
    test_matches = set(unique_matches[n_train + n_val:])

    train_indices = np.where(metadata["match_id"].isin(train_matches))[0]
    val_indices = np.where(metadata["match_id"].isin(val_matches))[0]
    test_indices = np.where(metadata["match_id"].isin(test_matches))[0]

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def create_dataloaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config: dict,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = StrokeDataset(
        sequences[train_indices],
        labels[train_indices],
        augment=True,
        config=config,
    )
    val_dataset = StrokeDataset(
        sequences[val_indices],
        labels[val_indices],
        augment=False,
        config=config,
    )

    bs = batch_size or config["training"]["lstm"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def load_processed_data(config: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load preprocessed sequences, labels, and metadata from disk."""
    processed_dir = Path(config["paths"]["processed"])

    sequences = np.load(processed_dir / "sequences.npy")
    labels = np.load(processed_dir / "labels.npy")
    metadata = pd.read_csv(processed_dir / "metadata.csv")

    return sequences, labels, metadata


if __name__ == "__main__":
    config = load_config()
    sequences, labels, metadata = load_processed_data(config)

    print(f"Loaded {len(sequences)} sequences, shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}, unique: {np.unique(labels)}")
    print(f"Matches: {metadata['match_id'].nunique()}")

    # Test LOSO
    folds = get_loso_splits(metadata, sequences, labels)
    print(f"\nLOSO folds: {len(folds)}")
    for i, fold in enumerate(folds[:3]):
        print(f"  Fold {i}: train={len(fold['train_indices'])}, val={len(fold['val_indices'])}")

    # Test stratified
    split = get_stratified_split(metadata, sequences, labels, config["data"]["stratified_ratios"])
    print(f"\nStratified split: "
          f"train={len(split['train_indices'])}, "
          f"val={len(split['val_indices'])}, "
          f"test={len(split['test_indices'])}")
