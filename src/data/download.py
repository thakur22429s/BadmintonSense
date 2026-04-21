"""Download ShuttleSet annotations from CoachAI-Projects repository."""

import os
import urllib.request
import urllib.parse
import json
from pathlib import Path

import yaml
import pandas as pd
from tqdm import tqdm


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


GITHUB_API = "https://api.github.com/repos/wywyWang/CoachAI-Projects/contents"
RAW_BASE = "https://raw.githubusercontent.com/wywyWang/CoachAI-Projects/main"


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def list_github_dir(api_path: str) -> list[dict]:
    """List contents of a GitHub directory via the API."""
    url = f"{GITHUB_API}/{api_path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  Failed to list {api_path}: {e}")
        return []


def download_shuttleset_annotations(output_dir: Path, version: str = "v1") -> list[Path]:
    """Download ShuttleSet annotation CSVs.

    The repo structure is:
        ShuttleSet/set/match.csv
        ShuttleSet/set/homography.csv
        ShuttleSet/set/<MatchFolder>/set1.csv, set2.csv, set3.csv

    Args:
        output_dir: Directory to save annotation files.
        version: "v1" for ShuttleSet, "v2" for ShuttleSet22.

    Returns:
        List of paths to downloaded files.
    """
    dataset_name = "ShuttleSet" if version == "v1" else "ShuttleSet22"
    subdir = output_dir / f"shuttleset_{version}"
    subdir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    # Download root-level files (match.csv, homography.csv)
    print(f"  Downloading root metadata...")
    for filename in ["match.csv", "homography.csv"]:
        dest = subdir / filename
        if dest.exists():
            print(f"  Already exists: {dest}")
            downloaded.append(dest)
            continue
        url = f"{RAW_BASE}/{dataset_name}/set/{filename}"
        if download_file(url, dest):
            downloaded.append(dest)

    # List match folders via GitHub API
    print(f"  Listing match folders...")
    entries = list_github_dir(f"{dataset_name}/set")
    match_dirs = [e for e in entries if e.get("type") == "dir"]
    print(f"  Found {len(match_dirs)} match folders")

    # Download set CSVs from each match folder
    for match_entry in tqdm(match_dirs, desc=f"Downloading {dataset_name} matches"):
        match_name = match_entry["name"]
        match_dir = subdir / "matches" / match_name
        match_dir.mkdir(parents=True, exist_ok=True)

        for set_num in [1, 2, 3]:
            filename = f"set{set_num}.csv"
            dest = match_dir / filename
            if dest.exists():
                downloaded.append(dest)
                continue
            encoded_name = urllib.parse.quote(match_name)
            url = f"{RAW_BASE}/{dataset_name}/set/{encoded_name}/{filename}"
            if download_file(url, dest):
                downloaded.append(dest)

    return downloaded


def parse_annotations(annotation_dir: Path) -> pd.DataFrame:
    """Parse ShuttleSet annotations into a unified stroke DataFrame.

    Reads match.csv for metadata, then loads all per-match set CSVs.
    Each set CSV has columns: rally, ball_round, time, frame_num, player,
    type (Chinese stroke name), hit_x, hit_y, landing_x, landing_y, etc.

    Returns DataFrame with columns including:
        match_id, match_name, set_num, rally, ball_round, player, type,
        frame_num, hit_x, hit_y, landing_x, landing_y, video_id
    """
    match_csv = annotation_dir / "match.csv"
    matches_dir = annotation_dir / "matches"

    if not match_csv.exists():
        raise FileNotFoundError(
            f"Missing match.csv in {annotation_dir}. Run download first."
        )
    if not matches_dir.exists():
        raise FileNotFoundError(
            f"Missing matches/ directory in {annotation_dir}. Run download first."
        )

    # Parse match metadata
    matches = pd.read_csv(match_csv)
    print(f"  Loaded {len(matches)} matches from match.csv")

    # Load all set CSVs
    all_strokes = []
    match_folders = sorted([d for d in matches_dir.iterdir() if d.is_dir()])

    for match_folder in tqdm(match_folders, desc="Parsing match annotations"):
        match_name = match_folder.name

        # Find matching row in match.csv by folder name
        match_row = matches[matches["video"] == match_name] if "video" in matches.columns else pd.DataFrame()
        if match_row.empty:
            match_row = matches[matches["id"] == match_name] if "id" in matches.columns else pd.DataFrame()

        match_id = match_row["id"].values[0] if not match_row.empty and "id" in match_row.columns else match_name
        video_id = match_row["url"].values[0] if not match_row.empty and "url" in match_row.columns else ""

        # Extract YouTube video ID from URL if present
        if isinstance(video_id, str) and "youtube.com" in video_id:
            video_id = video_id.split("v=")[-1].split("&")[0]
        elif isinstance(video_id, str) and "youtu.be" in video_id:
            video_id = video_id.split("/")[-1].split("?")[0]

        for set_csv in sorted(match_folder.glob("set*.csv")):
            set_num = int(set_csv.stem.replace("set", ""))
            try:
                df = pd.read_csv(set_csv)
            except Exception as e:
                print(f"  Error reading {set_csv}: {e}")
                continue

            df["match_id"] = match_id
            df["match_name"] = match_name
            df["set_num"] = set_num
            df["video_id"] = video_id
            all_strokes.append(df)

    if not all_strokes:
        raise ValueError("No stroke data found in any match folder.")

    combined = pd.concat(all_strokes, ignore_index=True)
    print(f"  Total strokes loaded: {len(combined)}")

    return combined


def extract_video_ids(annotations: pd.DataFrame) -> list[dict]:
    """Extract unique YouTube video IDs from annotation data."""
    unique = annotations[["video_id", "match_id", "match_name"]].drop_duplicates(subset=["match_id"])

    video_list = []
    for _, row in unique.iterrows():
        vid_id = row["video_id"]
        if not vid_id or pd.isna(vid_id) or vid_id == "":
            continue
        video_list.append({
            "video_id": vid_id,
            "match_id": row["match_id"],
            "match_name": row["match_name"],
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
        match_name = entry["match_name"]
        lines.append(f'echo "Downloading: {match_name} ({vid_id})..."')
        lines.append(
            f'yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" '
            f'--merge-output-format mp4 '
            f'-o "$OUTPUT_DIR/{match_name}.mp4" '
            f'--no-overwrites '
            f'"https://www.youtube.com/watch?v={vid_id}" || echo "  FAILED: {vid_id}"'
        )
        lines.append("")

    lines.append('echo "Download complete."')

    # Video mapping JSON: match_name -> filename
    mapping = {e["match_name"]: f"{e['match_name']}.mp4" for e in video_list}
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

    print("=" * 60)
    print("Downloading ShuttleSet v1 annotations")
    print("=" * 60)
    downloaded_v1 = download_shuttleset_annotations(ann_dir, version="v1")
    print(f"Downloaded {len(downloaded_v1)} files")

    print(f"\n{'='*60}")
    print("Parsing annotations")
    print("=" * 60)
    v1_dir = ann_dir / "shuttleset_v1"
    annotations = parse_annotations(v1_dir)
    print(f"Total strokes: {len(annotations)}")
    if "type" in annotations.columns:
        print(f"\nStroke types:\n{annotations['type'].value_counts().to_string()}")

    print(f"\n{'='*60}")
    print("Extracting video IDs")
    print("=" * 60)
    video_list = extract_video_ids(annotations)
    print(f"Unique videos: {len(video_list)}")

    print(f"\n{'='*60}")
    print("Generating download script")
    print("=" * 60)
    script_path = Path("scripts/02_download_videos.sh")
    video_dir = Path(config["paths"]["raw_videos"])
    generate_download_script(video_list, script_path, video_dir)
