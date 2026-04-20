"""Download ShuttleSet annotations from CoachAI-Projects repository."""

import os
import urllib.request
import json
from pathlib import Path

import yaml
import pandas as pd
from tqdm import tqdm


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


SHUTTLESET_BASE_URL = (
    "https://raw.githubusercontent.com/wywyWang/CoachAI-Projects/main/ShuttleSet"
)

SHUTTLESET22_BASE_URL = (
    "https://raw.githubusercontent.com/wywyWang/CoachAI-Projects/main/ShuttleSet22"
)

ANNOTATION_FILES = [
    "set/match.csv",
    "set/player.csv",
    "set/set.csv",
    "set/rally.csv",
    "set/ball_round.csv",
]

ANNOTATION_FILES_22 = [
    "set/match.csv",
    "set/player.csv",
    "set/set.csv",
    "set/rally.csv",
    "set/ball_round.csv",
]


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def download_shuttleset_annotations(output_dir: Path, version: str = "v1") -> list[Path]:
    """Download ShuttleSet annotation CSVs.

    Args:
        output_dir: Directory to save annotation files.
        version: "v1" for ShuttleSet (KDD 2023), "v2" for ShuttleSet22 (IJCAI 2024).

    Returns:
        List of paths to downloaded files.
    """
    base_url = SHUTTLESET_BASE_URL if version == "v1" else SHUTTLESET22_BASE_URL
    files = ANNOTATION_FILES if version == "v1" else ANNOTATION_FILES_22
    subdir = output_dir / f"shuttleset_{version}"
    subdir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for rel_path in tqdm(files, desc=f"Downloading ShuttleSet {version}"):
        url = f"{base_url}/{rel_path}"
        dest = subdir / Path(rel_path).name
        if dest.exists():
            print(f"  Already exists: {dest}")
            downloaded.append(dest)
            continue
        if download_file(url, dest):
            downloaded.append(dest)

    return downloaded


def parse_annotations(annotation_dir: Path) -> pd.DataFrame:
    """Parse ShuttleSet annotations into a unified stroke DataFrame.

    Returns DataFrame with columns:
        match_id, set_id, rally_id, ball_round, player, stroke_type,
        video_id, frame_num, landing_x, landing_y
    """
    match_path = annotation_dir / "match.csv"
    rally_path = annotation_dir / "rally.csv"
    ball_round_path = annotation_dir / "ball_round.csv"

    if not all(p.exists() for p in [match_path, rally_path, ball_round_path]):
        raise FileNotFoundError(
            f"Missing annotation files in {annotation_dir}. "
            "Run download first."
        )

    matches = pd.read_csv(match_path)
    rallies = pd.read_csv(rally_path)
    strokes = pd.read_csv(ball_round_path)

    # Merge to get full context per stroke
    merged = strokes.merge(
        rallies[["match_id", "set", "rally", "video", "rally_start_frame", "rally_end_frame"]],
        on=["match_id", "set", "rally"],
        how="left",
    )

    merged = merged.merge(
        matches[["match_id", "player_a", "player_b"]],
        on="match_id",
        how="left",
    )

    return merged


def extract_video_ids(annotations: pd.DataFrame) -> list[dict]:
    """Extract unique YouTube video IDs from annotation data.

    Returns list of dicts: {video_id, match_id, url}
    """
    video_col = "video" if "video" in annotations.columns else "video_id"
    unique_videos = annotations[[video_col, "match_id"]].drop_duplicates()

    video_list = []
    for _, row in unique_videos.iterrows():
        vid_id = row[video_col]
        video_list.append({
            "video_id": vid_id,
            "match_id": row["match_id"],
            "url": f"https://www.youtube.com/watch?v={vid_id}",
        })

    return video_list


def generate_download_script(video_list: list[dict], output_path: Path, video_dir: Path):
    """Generate a yt-dlp shell script for downloading BWF match videos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/bin/bash",
        "# Auto-generated: Download BWF match videos for ShuttleSet",
        "# Requires: yt-dlp (pip install yt-dlp)",
        "",
        f'OUTPUT_DIR="{video_dir}"',
        'mkdir -p "$OUTPUT_DIR"',
        "",
        "# Download each match video at 720p, 30fps",
    ]

    for entry in video_list:
        vid_id = entry["video_id"]
        match_id = entry["match_id"]
        lines.append(f'echo "Downloading match {match_id} ({vid_id})..."')
        lines.append(
            f'yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" '
            f'--merge-output-format mp4 '
            f'-o "$OUTPUT_DIR/{match_id}_{vid_id}.mp4" '
            f'--no-overwrites '
            f'"https://www.youtube.com/watch?v={vid_id}" || echo "  FAILED: {vid_id}"'
        )
        lines.append("")

    lines.append('echo "Download complete."')

    # Video mapping JSON
    mapping = {e["match_id"]: f"{e['match_id']}_{e['video_id']}.mp4" for e in video_list}
    mapping_path = output_path.parent / "video_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(lines))

    os.chmod(output_path, 0o755)
    print(f"Download script written to: {output_path}")
    print(f"Video mapping written to: {mapping_path}")


if __name__ == "__main__":
    config = load_config()
    ann_dir = Path(config["paths"]["annotations"])

    print("=== Downloading ShuttleSet v1 annotations ===")
    downloaded_v1 = download_shuttleset_annotations(ann_dir, version="v1")
    print(f"Downloaded {len(downloaded_v1)} files")

    print("\n=== Downloading ShuttleSet22 annotations ===")
    downloaded_v2 = download_shuttleset_annotations(ann_dir, version="v2")
    print(f"Downloaded {len(downloaded_v2)} files")

    print("\n=== Parsing annotations ===")
    v1_dir = ann_dir / "shuttleset_v1"
    annotations = parse_annotations(v1_dir)
    print(f"Total strokes: {len(annotations)}")
    if "type" in annotations.columns:
        print(f"Stroke types:\n{annotations['type'].value_counts()}")

    print("\n=== Extracting video IDs ===")
    video_list = extract_video_ids(annotations)
    print(f"Unique videos: {len(video_list)}")

    print("\n=== Generating download script ===")
    script_path = Path("scripts/02_download_videos.sh")
    video_dir = Path(config["paths"]["raw_videos"])
    generate_download_script(video_list, script_path, video_dir)
