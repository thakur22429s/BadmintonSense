"""Preprocessing: normalization, keypoint selection, temporal resampling, augmentation."""

from pathlib import Path

import numpy as np
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


MIRROR_PAIRS = {
    11: 12, 12: 11,  # shoulders
    13: 14, 14: 13,  # elbows
    15: 16, 16: 15,  # wrists
    17: 18, 18: 17,  # pinky
    19: 20, 20: 19,  # index
    23: 24, 24: 23,  # hips
    25: 26, 26: 25,  # knees
}


def select_keypoints(keypoints: np.ndarray, indices: list[int]) -> np.ndarray:
    """Select subset of keypoints.

    Args:
        keypoints: shape (T, 33, 3)
        indices: list of MediaPipe landmark indices to keep.

    Returns:
        shape (T, len(indices), 3)
    """
    return keypoints[:, indices, :]


def normalize_to_hip_center(keypoints: np.ndarray, hip_left_idx: int, hip_right_idx: int) -> np.ndarray:
    """Translate keypoints so hip midpoint is at origin per frame.

    Args:
        keypoints: shape (T, K, 3)
        hip_left_idx: index of left hip in the selected keypoint array.
        hip_right_idx: index of right hip in the selected keypoint array.

    Returns:
        Translated keypoints, shape (T, K, 3)
    """
    hip_center = (keypoints[:, hip_left_idx, :] + keypoints[:, hip_right_idx, :]) / 2.0
    return keypoints - hip_center[:, np.newaxis, :]


def scale_by_torso(
    keypoints: np.ndarray,
    shoulder_left_idx: int,
    shoulder_right_idx: int,
    hip_left_idx: int,
    hip_right_idx: int,
) -> np.ndarray:
    """Scale keypoints by torso length for size invariance.

    Torso length = average of (left_shoulder-left_hip, right_shoulder-right_hip) distances.
    """
    left_torso = np.linalg.norm(
        keypoints[:, shoulder_left_idx, :] - keypoints[:, hip_left_idx, :], axis=1
    )
    right_torso = np.linalg.norm(
        keypoints[:, shoulder_right_idx, :] - keypoints[:, hip_right_idx, :], axis=1
    )
    torso_length = (left_torso + right_torso) / 2.0
    torso_length = np.clip(torso_length, 1e-6, None)  # avoid division by zero

    scale = torso_length[:, np.newaxis, np.newaxis]
    return keypoints / scale


def temporal_resample(keypoints: np.ndarray, target_length: int) -> np.ndarray:
    """Resample temporal dimension to fixed length using linear interpolation.

    Args:
        keypoints: shape (T, K, D)
        target_length: desired number of frames.

    Returns:
        shape (target_length, K, D)
    """
    T, K, D = keypoints.shape
    if T == target_length:
        return keypoints

    old_indices = np.linspace(0, T - 1, T)
    new_indices = np.linspace(0, T - 1, target_length)

    resampled = np.zeros((target_length, K, D))
    for k in range(K):
        for d in range(D):
            resampled[:, k, d] = np.interp(new_indices, old_indices, keypoints[:, k, d])

    return resampled


def augment_temporal_jitter(keypoints: np.ndarray, max_shift: int = 3) -> np.ndarray:
    """Randomly shift temporal window."""
    T = keypoints.shape[0]
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return keypoints
    if shift > 0:
        return np.concatenate([keypoints[shift:], np.tile(keypoints[-1:], (shift, 1, 1))], axis=0)
    else:
        return np.concatenate([np.tile(keypoints[0:1], (-shift, 1, 1)), keypoints[:shift]], axis=0)


def augment_gaussian_noise(keypoints: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to keypoint coordinates."""
    noise = np.random.randn(*keypoints.shape) * std
    return keypoints + noise


def augment_horizontal_flip(keypoints: np.ndarray, selected_indices: list[int]) -> np.ndarray:
    """Mirror keypoints horizontally, swapping left/right pairs.

    Args:
        keypoints: shape (T, K, 3) — already using selected keypoint subset.
        selected_indices: original MediaPipe indices for the selected keypoints.
    """
    flipped = keypoints.copy()
    flipped[:, :, 0] = -flipped[:, :, 0]  # negate x

    # Build index mapping for the selected subset
    idx_map = {orig: pos for pos, orig in enumerate(selected_indices)}
    swap_pairs = []
    seen = set()
    for orig_idx in selected_indices:
        if orig_idx in MIRROR_PAIRS and orig_idx not in seen:
            partner = MIRROR_PAIRS[orig_idx]
            if partner in idx_map:
                swap_pairs.append((idx_map[orig_idx], idx_map[partner]))
                seen.add(orig_idx)
                seen.add(partner)

    for i, j in swap_pairs:
        flipped[:, [i, j], :] = flipped[:, [j, i], :]

    return flipped


def augment_temporal_scale(keypoints: np.ndarray, scale_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """Speed up or slow down the sequence."""
    scale = np.random.uniform(*scale_range)
    T = keypoints.shape[0]
    new_T = int(T * scale)
    new_T = max(new_T, 2)
    resampled = temporal_resample(keypoints, new_T)
    return temporal_resample(resampled, T)


def augment_frame_dropout(keypoints: np.ndarray, dropout_rate: float = 0.15) -> np.ndarray:
    """Randomly drop frames and interpolate."""
    T, K, D = keypoints.shape
    n_drop = int(T * dropout_rate)
    if n_drop == 0:
        return keypoints

    keep_indices = sorted(np.random.choice(T, T - n_drop, replace=False))
    kept = keypoints[keep_indices]
    return temporal_resample(kept, T)


def preprocess_single(
    raw_keypoints: np.ndarray,
    config: dict,
    augment: bool = False,
) -> np.ndarray:
    """Full preprocessing pipeline for a single pose sequence.

    Args:
        raw_keypoints: shape (T, 33, 3) from MediaPipe.
        config: Configuration dict.
        augment: Whether to apply data augmentation.

    Returns:
        Preprocessed keypoints, shape (target_length, num_selected_keypoints, 3)
    """
    selected_indices = config["data"]["selected_keypoints"]
    target_length = config["data"]["sequence_length"]
    aug_config = config["data"]["augmentation"]

    # Select upper-body keypoints
    kps = select_keypoints(raw_keypoints, selected_indices)

    # Find hip and shoulder indices in the selected subset
    idx_map = {orig: pos for pos, orig in enumerate(selected_indices)}
    hip_left_pos = idx_map[23]
    hip_right_pos = idx_map[24]
    shoulder_left_pos = idx_map[11]
    shoulder_right_pos = idx_map[12]

    # Normalize
    if config["data"]["normalize_to_hip"]:
        kps = normalize_to_hip_center(kps, hip_left_pos, hip_right_pos)

    if config["data"]["scale_by_torso"]:
        kps = scale_by_torso(kps, shoulder_left_pos, shoulder_right_pos, hip_left_pos, hip_right_pos)

    # Temporal resample to fixed length
    kps = temporal_resample(kps, target_length)

    # Augmentation (training only)
    if augment and aug_config["enabled"]:
        if np.random.random() < 0.5:
            kps = augment_temporal_jitter(kps, aug_config["temporal_jitter_frames"])
        kps = augment_gaussian_noise(kps, aug_config["gaussian_noise_std"])
        if np.random.random() < aug_config["horizontal_flip_prob"]:
            kps = augment_horizontal_flip(kps, selected_indices)
        if np.random.random() < 0.3:
            kps = augment_temporal_scale(kps, tuple(aug_config["temporal_scale_range"]))
        if np.random.random() < 0.3:
            kps = augment_frame_dropout(kps, aug_config["frame_dropout_rate"])

    return kps


def preprocess_dataset(
    pose_metadata: "pd.DataFrame",
    output_dir: Path,
    config: dict,
    class_mapping: dict,
) -> None:
    """Preprocess all valid poses and save to processed directory.

    Saves:
        - sequences.npy: shape (N, T, K, 3)
        - labels.npy: shape (N,) integer class indices
        - metadata.csv: clip_id, match_id, label, original_type
    """
    import pandas as pd
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)

    valid = pose_metadata[pose_metadata["pose_valid"] == True].copy()
    print(f"Processing {len(valid)} valid pose sequences...")

    class_names = config["classes"]["names"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    sequences = []
    labels = []
    meta_records = []

    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="Preprocessing"):
        pose_path = Path(row["pose_path"])
        if not pose_path.exists():
            continue

        data = np.load(pose_path, allow_pickle=True).item()
        raw_kps = data["keypoints"]

        if np.isnan(raw_kps).all():
            continue

        # Map stroke type to our class taxonomy (Chinese labels from ShuttleSet)
        original_type = str(row.get("stroke_type", "other")).strip()
        mapped_class = class_mapping.get(original_type, "Other")
        if mapped_class not in class_to_idx:
            mapped_class = "Other"
        label_idx = class_to_idx[mapped_class]

        processed = preprocess_single(raw_kps, config, augment=False)
        sequences.append(processed)
        labels.append(label_idx)
        meta_records.append({
            "clip_id": row["clip_id"],
            "match_id": row["match_id"],
            "label": label_idx,
            "class_name": mapped_class,
            "original_type": original_type,
        })

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    np.save(output_dir / "sequences.npy", sequences)
    np.save(output_dir / "labels.npy", labels)
    pd.DataFrame(meta_records).to_csv(output_dir / "metadata.csv", index=False)

    print(f"Saved {len(sequences)} sequences to {output_dir}")
    print(f"Shape: {sequences.shape}")
    print(f"Class distribution:")
    for i, name in enumerate(class_names):
        count = (labels == i).sum()
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")


if __name__ == "__main__":
    import pandas as pd

    config = load_config()
    pose_dir = Path(config["paths"]["poses"])
    processed_dir = Path(config["paths"]["processed"])
    class_mapping = config["classes"]["shuttleset_mapping"]

    metadata_path = pose_dir / "pose_metadata.csv"
    if not metadata_path.exists():
        print("Run pose extraction first (04_extract_poses.py)")
        exit(1)

    pose_metadata = pd.read_csv(metadata_path)
    preprocess_dataset(pose_metadata, processed_dir, config, class_mapping)
