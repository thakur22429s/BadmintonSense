"""Step 5: Normalize keypoints, resample, and create train/val/test splits."""

import sys
sys.path.insert(0, ".")

from pathlib import Path
import pandas as pd
from src.data.download import load_config
from src.data.preprocessing import preprocess_dataset


def main():
    config = load_config()

    print("=" * 60)
    print("STEP 5: Preprocess & Split Data")
    print("=" * 60)

    pose_dir = Path(config["paths"]["poses"])
    processed_dir = Path(config["paths"]["processed"])
    class_mapping = config["classes"]["shuttleset_mapping"]

    metadata_path = pose_dir / "pose_metadata.csv"
    if not metadata_path.exists():
        print("ERROR: No pose metadata found. Run 04_extract_poses.py first.")
        return

    pose_metadata = pd.read_csv(metadata_path)
    print(f"  Total poses: {len(pose_metadata)}")
    print(f"  Valid poses: {pose_metadata['pose_valid'].sum()}")
    print(f"  Sequence length: {config['data']['sequence_length']}")
    print(f"  Selected keypoints: {config['data']['num_keypoints']}")

    print("\nPreprocessing...")
    preprocess_dataset(pose_metadata, processed_dir, config, class_mapping)

    print(f"\n{'='*60}")
    print(f"Processed data saved to: {processed_dir}")
    print(f"Next: python scripts/06_train.py")


if __name__ == "__main__":
    main()
