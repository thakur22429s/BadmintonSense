"""Extract 2D/3D pose keypoints from stroke clips using MediaPipe Tasks API."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# Map model_complexity (0=lite, 1=full, 2=heavy) to task file names
_MODEL_FILES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


def _get_model_path(model_complexity: int = 2) -> str:
    """Resolve the MediaPipe pose landmarker .task model file path."""
    filename = _MODEL_FILES.get(model_complexity, _MODEL_FILES[2])
    # Check local models/ directory first
    local_path = Path("models") / filename
    if local_path.exists():
        return str(local_path)
    # Check project root
    project_root = Path(__file__).resolve().parent.parent.parent
    root_path = project_root / "models" / filename
    if root_path.exists():
        return str(root_path)
    raise FileNotFoundError(
        f"MediaPipe model not found: {filename}. "
        f"Download from: https://storage.googleapis.com/mediapipe-models/"
        f"pose_landmarker/pose_landmarker_heavy/float16/latest/"
        f"pose_landmarker_heavy.task and place in models/ directory."
    )


def _create_landmarker(
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create a reusable PoseLandmarker instance in IMAGE mode.

    IMAGE mode processes each frame independently — no timestamp ordering
    required, which allows reusing one landmarker across multiple clips.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    model_path = _get_model_path(model_complexity)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return PoseLandmarker.create_from_options(options)


def extract_poses_from_clip(
    clip_path: Path,
    landmarker=None,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pose keypoints from a video clip using MediaPipe Tasks API.

    Args:
        clip_path: Path to the video clip.
        landmarker: Reusable PoseLandmarker instance. If None, creates a new one.
        model_complexity: 0=lite, 1=full, 2=heavy (only used if landmarker is None).
        min_detection_confidence: Minimum detection confidence.
        min_tracking_confidence: Minimum tracking confidence.

    Returns:
        Tuple of (keypoints, confidences):
            keypoints: shape (T, 33, 3) -- x, y, z per landmark per frame
            confidences: shape (T, 33) -- visibility per landmark per frame
    """
    import mediapipe as mp

    own_landmarker = False
    if landmarker is None:
        landmarker = _create_landmarker(
            model_complexity, min_detection_confidence, min_tracking_confidence
        )
        own_landmarker = True

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        if own_landmarker:
            landmarker.close()
        raise IOError(f"Cannot open video: {clip_path}")

    all_keypoints = []
    all_confidences = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                frame_kps = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                frame_conf = np.array([lm.visibility for lm in landmarks])
            else:
                frame_kps = np.full((33, 3), np.nan)
                frame_conf = np.zeros(33)

            all_keypoints.append(frame_kps)
            all_confidences.append(frame_conf)
    finally:
        cap.release()
        if own_landmarker:
            landmarker.close()

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

    Creates one PoseLandmarker and reuses it across all clips for efficiency.

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

    # Create landmarker once for all clips
    print(f"  Loading MediaPipe PoseLandmarker (complexity={model_complexity})...")
    landmarker = _create_landmarker(model_complexity, min_det, min_track)

    pose_paths = []
    detection_rates = []
    valid_mask = []

    try:
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
                    clip_path, landmarker=landmarker
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
    finally:
        landmarker.close()

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
