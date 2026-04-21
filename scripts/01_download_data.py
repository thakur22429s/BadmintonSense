"""Step 1: Download ShuttleSet annotations from CoachAI-Projects."""

import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.data.download import (
    load_config,
    download_shuttleset_annotations,
    parse_annotations,
    extract_video_ids,
    generate_download_script,
)


def main():
    config = load_config()
    ann_dir = Path(config["paths"]["annotations"])

    print("=" * 60)
    print("STEP 1: Download ShuttleSet Annotations")
    print("=" * 60)

    print("\n[1/3] Downloading ShuttleSet v1 (KDD 2023)...")
    downloaded_v1 = download_shuttleset_annotations(ann_dir, version="v1")
    print(f"  Downloaded {len(downloaded_v1)} files")

    print("\n[2/3] Parsing annotations...")
    v1_dir = ann_dir / "shuttleset_v1"
    try:
        annotations = parse_annotations(v1_dir)
        print(f"  Total strokes: {len(annotations)}")
        if "type" in annotations.columns:
            print(f"  Stroke types found: {annotations['type'].nunique()}")
            try:
                print(f"  Distribution:\n{annotations['type'].value_counts().to_string()}")
            except UnicodeEncodeError:
                # Windows console can't render Chinese chars — write to file instead
                dist = annotations['type'].value_counts()
                dist_path = v1_dir / "stroke_distribution.csv"
                dist.to_csv(dist_path, encoding="utf-8-sig")
                print(f"  (Distribution saved to {dist_path} — console can't render Chinese chars)")
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        print("  Annotations may need manual download from CoachAI GitHub.")
        return

    print("\n[3/3] Generating video download script...")
    video_list = extract_video_ids(annotations)
    print(f"  Unique videos to download: {len(video_list)}")

    script_path = Path("scripts/02_download_videos.sh")
    video_dir = Path(config["paths"]["raw_videos"])
    generate_download_script(video_list, script_path, video_dir)

    # Save annotations for downstream use
    ann_save_path = v1_dir / "all_strokes.csv"
    annotations.to_csv(ann_save_path, index=False, encoding="utf-8-sig")
    print(f"  Saved combined annotations to: {ann_save_path}")

    print("\n" + "=" * 60)
    print("DONE. Next steps:")
    print("  1. Review scripts/02_download_videos.sh")
    print("  2. Run: bash scripts/02_download_videos.sh")
    print("  3. Then run: python scripts/03_extract_clips.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
