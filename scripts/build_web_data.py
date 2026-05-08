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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def load_models(config) -> tuple[dict, "torch.device"]:
    device = get_device()
    out = {}
    for mtype, builder, ckpt_name in (
        ("lstm", build_lstm, "best_lstm_stratified.pt"),
        ("transformer", build_transformer, "best_transformer_stratified.pt"),
    ):
        ckpt_path = Path(config["paths"]["models"]) / ckpt_name
        if not ckpt_path.exists():
            print(f"WARNING: missing checkpoint {ckpt_path} -- skipping {mtype}")
            continue
        model = builder(config)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        out[mtype] = model
    return out, device


def predict_probs(model, device, config, kps):
    if len(kps) == 0:
        return None
    try:
        processed = preprocess_single(kps, config, augment=False)
    except Exception:
        return None
    if np.isnan(processed).all():
        return None
    processed = np.nan_to_num(processed, nan=0.0)
    T, K, D = processed.shape
    flat = processed.reshape(1, T, K * D)
    x = torch.tensor(flat, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    if np.isnan(probs).any():
        return None
    return probs


def emit_match_json(
    match_name: str,
    match_df: pd.DataFrame,
    models: dict,
    device,
    config,
    match_info: dict,
    out_dir: Path,
    keypoint_subset_n: int = 50,
) -> dict:
    class_names = config["classes"]["names"]
    class_mapping = config["classes"]["shuttleset_mapping"]

    # Decide which strokes get keypoints baked into JSON: spread by class for /single-clip
    valid_df = match_df[match_df["valid"] == True].reset_index(drop=True)
    chosen_kp_ids = set()
    if len(valid_df):
        per_class = max(1, keypoint_subset_n // max(len(class_names), 1))
        for c in class_names:
            mapped = valid_df[valid_df["stroke_type"].map(class_mapping).fillna("Other") == c]
            chosen_kp_ids.update(mapped["clip_id"].head(per_class).tolist())

    strokes_out = []
    preds_by_model = {m: {} for m in models}

    for _, row in valid_df.iterrows():
        try:
            pose_data = np.load(row["pose_path"], allow_pickle=True).item()
            kps = pose_data["keypoints"]
        except Exception:
            continue

        truth = class_mapping.get(row["stroke_type"], "Other")
        truth_idx = class_names.index(truth) if truth in class_names else -1

        keypoints_field = None
        if row["clip_id"] in chosen_kp_ids:
            keypoints_field = np.nan_to_num(kps, nan=0.0).round(4).tolist()

        for mname, model in models.items():
            probs = predict_probs(model, device, config, kps)
            if probs is None:
                preds_by_model[mname][row["clip_id"]] = None
                continue
            pred_idx = int(np.argmax(probs))
            preds_by_model[mname][row["clip_id"]] = {
                "predIdx": pred_idx,
                "probs": [round(float(p), 4) for p in probs],
                "correct": (truth_idx == pred_idx),
            }

        strokes_out.append({
            "clipId": row["clip_id"],
            "setNum": int(row["set_num"]),
            "rallyNum": int(row["rally_num"]),
            "ballNum": float(row["ball_num"]),
            "player": row.get("player", ""),
            "playerName": player_real_name(match_name, row.get("player", ""), match_info),
            "truthClass": truth,
            "truthIdx": truth_idx,
            "detectionRate": round(float(row.get("detection_rate", 0.0)), 4),
            "clipUrl": f"/clips/{row['clip_id']}.mp4",
            "keypoints": keypoints_field,
        })

    # Confusion + per-class F1 per model
    model_blocks = {}
    n_classes = len(class_names)
    for mname, preds in preds_by_model.items():
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        per_class_correct = np.zeros(n_classes, dtype=int)
        per_class_pred_total = np.zeros(n_classes, dtype=int)
        per_class_truth_total = np.zeros(n_classes, dtype=int)

        for s in strokes_out:
            p = preds.get(s["clipId"])
            t_idx = s["truthIdx"]
            if p is None or t_idx < 0:
                continue
            confusion[t_idx, p["predIdx"]] += 1
            per_class_pred_total[p["predIdx"]] += 1
            per_class_truth_total[t_idx] += 1
            if p["predIdx"] == t_idx:
                per_class_correct[t_idx] += 1

        per_class_f1 = {}
        for i, c in enumerate(class_names):
            tp = per_class_correct[i]
            prec = tp / per_class_pred_total[i] if per_class_pred_total[i] else 0.0
            rec = tp / per_class_truth_total[i] if per_class_truth_total[i] else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class_f1[c] = round(float(f1), 4)

        model_blocks[mname] = {
            "byClipId": {k: v for k, v in preds.items() if v is not None},
            "confusion": confusion.tolist(),
            "perClassF1": per_class_f1,
        }

    info = match_info.get(match_name, {})
    return {
        "slug": slugify(match_name),
        "rawName": match_name,
        "display": pretty_match_name(match_name, match_info),
        "winner": info.get("winner", "?"),
        "loser": info.get("loser", "?"),
        "tournament": info.get("tournament", "?"),
        "year": info.get("year", 0),
        "round": info.get("round", "?"),
        "classNames": class_names,
        "models": model_blocks,
        "strokes": strokes_out,
    }


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

    models, device = load_models(config)
    if not models:
        print("FATAL: no checkpoints loaded")
        return 2

    for entry in manifest:
        m_df = merged[merged["match_name"] == entry["rawName"]]
        match_json = emit_match_json(
            entry["rawName"], m_df, models, device, config, match_info, out_data
        )
        target = out_data / "matches" / f"{entry['slug']}.json"
        target.write_text(json.dumps(match_json))
        print(f"  emitted {target.name}: {len(match_json['strokes'])} strokes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
