"""Pre-compute every prediction the web showcase needs.

Reads existing checkpoints + cached pose .npy files, runs both BiLSTM and
Transformer over every valid clip, and writes JSON + curated MP4s into
web/public/.

Run once locally before `npm run build`:

    python scripts/build_web_data.py --out web/public

Idempotent: re-running overwrites the output directory.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")

from src.data.download import load_config
from src.data.preprocessing import preprocess_single
from src.models.lstm import build_lstm
from src.models.transformer import build_transformer
from src.models.utils import get_device

CLIP_PATTERN = re.compile(r"_s(\d+)_r(\d+)_b([\d.]+)$")


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def parse_clip_id(cid: str) -> tuple[str, int, int, float]:
    m = CLIP_PATTERN.search(cid)
    if not m:
        return cid, 0, 0, 0.0
    return cid[: m.start()], int(m.group(1)), int(m.group(2)), float(m.group(3))


def load_match_info(annotations_dir: Path) -> dict:
    p = annotations_dir / "shuttleset_v1" / "match.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {
        row["video"]: {
            "tournament": row["tournament"],
            "round": row["round"],
            "year": int(row["year"]),
            "winner": row["winner"],
            "loser": row["loser"],
        }
        for _, row in df.iterrows()
    }


def pretty_match_name(name: str, info: dict) -> str:
    i = info.get(name)
    if not i:
        return name.replace("_", " ")
    return f"{i['winner']} def. {i['loser']} — {i['tournament']} ({i['year']}) — {i['round']}"


def player_real_name(match_name: str, ab: str, info: dict) -> str:
    i = info.get(match_name)
    if not i:
        return f"Player {ab}"
    winner_first = match_name.lower().replace(" ", "").startswith(
        i["winner"].lower().replace(" ", "").replace("_", "")[:5]
    )
    if ab == "A":
        return i["winner"] if winner_first else i["loser"]
    return i["loser"] if winner_first else i["winner"]


def build_manifest(merged: pd.DataFrame, match_info: dict) -> list[dict]:
    out = []
    for match_name, group in merged.groupby("match_name"):
        info = match_info.get(match_name, {})
        out.append({
            "slug": slugify(match_name),
            "rawName": match_name,
            "display": pretty_match_name(match_name, match_info),
            "winner": info.get("winner", "?"),
            "loser": info.get("loser", "?"),
            "tournament": info.get("tournament", "?"),
            "round": info.get("round", "?"),
            "year": info.get("year", 0),
            "totalStrokes": int(len(group)),
            "validStrokes": int(group["valid"].sum()),
        })
    return sorted(out, key=lambda r: (r["year"], r["display"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("web/public"))
    args = parser.parse_args()

    config = load_config()
    out_data = args.out / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    (out_data / "matches").mkdir(parents=True, exist_ok=True)

    clip_meta = pd.read_csv("data/clips/clip_metadata.csv")
    pose_meta = pd.read_csv("data/poses/pose_metadata.csv")
    valid_col = "pose_valid" if "pose_valid" in pose_meta.columns else "valid"
    cols = ["clip_id", "pose_path", "detection_rate"]
    if valid_col in pose_meta.columns:
        cols.append(valid_col)
    merged = clip_meta.merge(pose_meta[cols], on="clip_id", how="left")
    if valid_col in merged.columns:
        merged = merged.rename(columns={valid_col: "valid"})
    else:
        merged["valid"] = merged["detection_rate"].fillna(0) >= 0.5

    parsed = merged["clip_id"].apply(parse_clip_id)
    merged["match_name"] = [p[0] for p in parsed]
    merged["set_num"] = [p[1] for p in parsed]
    merged["rally_num"] = [p[2] for p in parsed]
    merged["ball_num"] = [p[3] for p in parsed]

    match_info = load_match_info(Path(config["paths"]["annotations"]))
    manifest = build_manifest(merged, match_info)

    (out_data / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest.json written: {len(manifest)} matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
