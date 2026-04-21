"""Segment full match videos into individual stroke clips using annotations."""

import subprocess
import json
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def frame_to_timestamp(frame: int, fps: int = 30) -> str:
    seconds = frame / fps
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: int = 30,
) -> bool:
    """Extract a clip from a video using ffmpeg.

    Args:
        video_path: Path to source video.
        output_path: Where to save the clip.
        start_frame: First frame of the clip.
        end_frame: Last frame of the clip.
        fps: Video framerate.

    Returns:
        True if extraction succeeded.
    """
    start_time = frame_to_timestamp(start_frame, fps)
    duration = (end_frame - start_frame) / fps

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", start_time,
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-an",  # no audio
        "-r", str(fps),
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ffmpeg error: {e}")
        return False


def extract_all_clips(
    annotations: pd.DataFrame,
    video_dir: Path,
    output_dir: Path,
    config: dict,
) -> pd.DataFrame:
    """Extract stroke clips for all annotated strokes.

    Args:
        annotations: DataFrame with stroke annotations (needs frame_num, match_id, video columns).
        video_dir: Directory containing downloaded match videos.
        output_dir: Where to save extracted clips.
        config: Configuration dict.

    Returns:
        DataFrame with clip metadata (clip_id, path, stroke_type, match_id, etc.)
    """
    fps = config["data"]["clip_fps"]
    pre_frames = int(config["data"]["clip_pre_hit_seconds"] * fps)
    post_frames = int(config["data"]["clip_post_hit_seconds"] * fps)

    # Load video mapping
    mapping_path = Path("scripts/video_mapping.json")
    if mapping_path.exists():
        with open(mapping_path) as f:
            video_mapping = json.load(f)
    else:
        video_mapping = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_metadata = []
    failed = 0

    frame_col = _find_frame_column(annotations)
    type_col = _find_type_column(annotations)

    for idx, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Extracting clips"):
        match_id = str(row["match_id"])
        match_name = str(row.get("match_name", match_id))

        hit_frame_raw = row.get(frame_col, None)
        if pd.isna(hit_frame_raw):
            failed += 1
            continue
        hit_frame = int(hit_frame_raw)

        # Determine video file path (videos are named by match_name)
        if match_name in video_mapping:
            video_file = video_dir / video_mapping[match_name]
        elif match_id in video_mapping:
            video_file = video_dir / video_mapping[match_id]
        else:
            video_file = _find_video_file(video_dir, match_name)

        if video_file is None or not video_file.exists():
            failed += 1
            continue

        start_frame = max(0, hit_frame - pre_frames)
        end_frame = hit_frame + post_frames

        set_num = row.get("set_num", 0)
        clip_id = f"{match_name}_s{set_num}_r{row.get('rally', 0)}_b{row.get('ball_round', idx)}"
        clip_path = output_dir / f"{clip_id}.mp4"

        if clip_path.exists():
            clip_metadata.append(_build_clip_record(clip_id, clip_path, row, type_col, match_id))
            continue

        success = extract_clip(video_file, clip_path, start_frame, end_frame, fps)
        if success:
            clip_metadata.append(_build_clip_record(clip_id, clip_path, row, type_col, match_id))
        else:
            failed += 1

    print(f"Extracted {len(clip_metadata)} clips, {failed} failures")

    df = pd.DataFrame(clip_metadata)
    metadata_path = output_dir / "clip_metadata.csv"
    df.to_csv(metadata_path, index=False)
    return df


def _find_frame_column(df: pd.DataFrame) -> str:
    for col in ["hit_frame", "frame_num", "frame", "start_frame"]:
        if col in df.columns:
            return col
    raise ValueError(f"No frame column found. Available: {list(df.columns)}")


def _find_type_column(df: pd.DataFrame) -> str:
    for col in ["type", "stroke_type", "shot_type", "ball_type"]:
        if col in df.columns:
            return col
    return "type"


def _find_video_file(video_dir: Path, match_id: str) -> Path | None:
    candidates = list(video_dir.glob(f"{match_id}*"))
    if candidates:
        return candidates[0]
    candidates = list(video_dir.glob(f"*{match_id}*"))
    return candidates[0] if candidates else None


def _build_clip_record(
    clip_id: str, clip_path: Path, row: pd.Series, type_col: str, match_id: str
) -> dict:
    return {
        "clip_id": clip_id,
        "clip_path": str(clip_path),
        "stroke_type": row.get(type_col, "unknown"),
        "match_id": match_id,
        "rally_id": row.get("rally", 0),
        "ball_round": row.get("ball_round", 0),
        "player": row.get("player", "unknown"),
    }


if __name__ == "__main__":
    from src.data.download import parse_annotations

    config = load_config()
    ann_dir = Path(config["paths"]["annotations"]) / "shuttleset_v1"
    annotations = parse_annotations(ann_dir)

    video_dir = Path(config["paths"]["raw_videos"])
    clip_dir = Path(config["paths"]["clips"])

    clips_df = extract_all_clips(annotations, video_dir, clip_dir, config)
    print(f"\nClip distribution:\n{clips_df['stroke_type'].value_counts()}")
