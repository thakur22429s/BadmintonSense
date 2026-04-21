"""Extract 2D/3D pose keypoints from stroke clips using MediaPipe."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_poses_from_clip(
    clip_path: Path,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pose keypoints from a video clip using MediaPipe.

    Args:
        clip_path: Path to the video clip.
        model_complexity: MediaPipe model complexity (0, 1, or 2).
        min_detection_confidence: Minimum detection confidence.
        min_tracking_confidence: Minimum tracking confidence.

    Returns:
        Tuple of (keypoints, confidences):
            keypoints: shape (T, 33, 3) — x, y, z per landmark per frame
            confidences: shape (T, 33) — visibility per landmark per frame
    """
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {clip_path}")

    all_keypoints = []
    all_confidences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_kps = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            frame_conf = np.array([lm.visibility for lm in landmarks])
        else:
            frame_kps = np.full((33, 3), np.nan)
            frame_conf = np.zeros(33)

        all_keypoints.append(frame_kps)
        all_confidences.append(frame_conf)

    cap.release()
    pose.close()

    return np.array(all_keypoints), np.array(all_confidences)


def compute_detection_rate(confidences: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of frames where pose was detected with sufficient confidence."""
    if len(confidences) == 0:
        return 0.0
    valid_frames = np.any(confidences > threshold, axis=1)
    return valid_frames.mean()


def interpolate_missing_frames(keypoints: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN frames in keypoint sequence.

    Args:
        keypoints: shape (T, K, 3), may contain NaN for missing detections.

    Returns:
        Interpolated keypoints with NaN replaced where possible.
    """
    T, K, D = keypoints.shape
    result = keypoints.copy()

    for k in range(K):
        for d in range(D):
            series = result[:, k, d]
            nans = np.isnan(series)
            if nans.all() or not nans.any():
                continue
            valid_idx = np.where(~nans)[0]
            result[:, k, d] = np.interp(
                np.arange(T), valid_idx, series[valid_idx]
            )

    return result


def extract_all_poses(
    clip_metadata: pd.DataFrame,
    output_dir: Path,
    config: dict,
) -> pd.DataFrame:
    """Run pose extraction on all clips.

    Args:
        clip_metadata: DataFrame with clip_id and clip_path columns.
        output_dir: Directory to save .npy pose files.
        config: Configuration dict.

    Returns:
        Updated metadata with pose_path and detection_rate columns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_complexity = config["data"]["mediapipe_model_complexity"]
    min_det = config["data"]["mediapipe_min_detection_confidence"]
    min_track = config["data"]["mediapipe_min_tracking_confidence"]
    min_rate = config["data"]["min_detection_rate"]

    pose_paths = []
    detection_rates = []
    valid_mask = []

    for _, row in tqdm(clip_metadata.iterrows(), total=len(clip_metadata), desc="Extracting poses"):
        clip_path = Path(row["clip_path"])
        clip_id = row["clip_id"]
        pose_path = output_dir / f"{clip_id}.npy"

        if pose_path.exists():
            data = np.load(pose_path, allow_pickle=True).item()
            rate = data.get("detection_rate", 1.0)
            pose_paths.append(str(pose_path))
            detection_rates.append(rate)
            valid_mask.append(rate >= min_rate)
            continue

        if not clip_path.exists():
            pose_paths.append("")
            detection_rates.append(0.0)
            valid_mask.append(False)
            continue

        try:
            keypoints, confidences = extract_poses_from_clip(
                clip_path, model_complexity, min_det, min_track
            )
        except Exception as e:
            print(f"  Failed on {clip_id}: {e}")
            pose_paths.append("")
            detection_rates.append(0.0)
            valid_mask.append(False)
            continue

        rate = compute_detection_rate(confidences)
        keypoints = interpolate_missing_frames(keypoints)

        np.save(pose_path, {
            "keypoints": keypoints,
            "confidences": confidences,
            "detection_rate": rate,
            "clip_id": clip_id,
            "num_frames": len(keypoints),
        })

        pose_paths.append(str(pose_path))
        detection_rates.append(rate)
        valid_mask.append(rate >= min_rate)

    result = clip_metadata.copy()
    result["pose_path"] = pose_paths
    result["detection_rate"] = detection_rates
    result["pose_valid"] = valid_mask

    n_valid = sum(valid_mask)
    print(f"Valid poses: {n_valid}/{len(clip_metadata)} "
          f"({n_valid/len(clip_metadata)*100:.1f}%)")
    print(f"Mean detection rate: {np.mean(detection_rates):.3f}")

    return result


if __name__ == "__main__":
    config = load_config()
    clip_dir = Path(config["paths"]["clips"])
    pose_dir = Path(config["paths"]["poses"])

    metadata_path = clip_dir / "clip_metadata.csv"
    if not metadata_path.exists():
        print("Run clip extraction first (03_extract_clips.py)")
        exit(1)

    clip_metadata = pd.read_csv(metadata_path)
    result = extract_all_poses(clip_metadata, pose_dir, config)
    result.to_csv(pose_dir / "pose_metadata.csv", index=False)
