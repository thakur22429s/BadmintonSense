"""Step 3: Segment match videos into individual stroke clips."""

import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.data.download import load_config, parse_annotations
from src.data.clip_extractor import extract_all_clips


def main():
    config = load_config()

    print("=" * 60)
    print("STEP 3: Extract Stroke Clips")
    print("=" * 60)

    ann_dir = Path(config["paths"]["annotations"]) / "shuttleset_v1"
    video_dir = Path(config["paths"]["raw_videos"])
    clip_dir = Path(config["paths"]["clips"])

    print(f"\nAnnotations: {ann_dir}")
    print(f"Videos: {video_dir}")
    print(f"Output: {clip_dir}")

    if not video_dir.exists() or not list(video_dir.glob("*.mp4")):
        print("\nERROR: No videos found. Run scripts/02_download_videos.sh first.")
        return

    print("\nParsing annotations...")
    annotations = parse_annotations(ann_dir)
    print(f"  Total strokes in annotations: {len(annotations)}")

    print("\nExtracting clips...")
    clips_df = extract_all_clips(annotations, video_dir, clip_dir, config)

    print(f"\n{'='*60}")
    print(f"Extracted {len(clips_df)} clips")
    print(f"Stroke distribution:")
    print(clips_df["stroke_type"].value_counts().to_string())
    print(f"\nNext: python scripts/04_extract_poses.py")


if __name__ == "__main__":
    main()
