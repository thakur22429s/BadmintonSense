"""Step 4: Run MediaPipe pose extraction on all stroke clips."""

import sys
sys.path.insert(0, ".")

from pathlib import Path
import pandas as pd
from src.data.download import load_config
from src.data.pose_extractor import extract_all_poses


def main():
    config = load_config()

    print("=" * 60)
    print("STEP 4: Extract Poses (MediaPipe)")
    print("=" * 60)

    clip_dir = Path(config["paths"]["clips"])
    pose_dir = Path(config["paths"]["poses"])

    metadata_path = clip_dir / "clip_metadata.csv"
    if not metadata_path.exists():
        print("ERROR: No clip metadata found. Run 03_extract_clips.py first.")
        return

    clip_metadata = pd.read_csv(metadata_path)
    print(f"  Clips to process: {len(clip_metadata)}")
    print(f"  MediaPipe complexity: {config['data']['mediapipe_model_complexity']}")
    print(f"  Min detection rate: {config['data']['min_detection_rate']}")

    print("\nExtracting poses...")
    result = extract_all_poses(clip_metadata, pose_dir, config)

    result.to_csv(pose_dir / "pose_metadata.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {pose_dir}/pose_metadata.csv")
    print(f"Next: python scripts/05_preprocess.py")


if __name__ == "__main__":
    main()
