"""Badminton-Sense — clean Streamlit demo for stroke classification + match analytics."""

import sys
sys.path.insert(0, ".")

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data.download import load_config
from src.data.pose_extractor import extract_poses_from_clip
from src.data.preprocessing import preprocess_single
from src.models.lstm import build_lstm
from src.models.transformer import build_transformer
from src.models.utils import get_device


# ============================================================
# PAGE CONFIG + STYLES
# ============================================================
st.set_page_config(
    page_title="Badminton-Sense",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f1729 0%, #1a2547 100%);
    }
    .main-title {
        font-family: 'Georgia', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    .main-subtitle {
        font-family: 'Calibri', sans-serif;
        font-size: 1.15rem;
        color: #cadcfc;
        font-style: italic;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .accent-bar {
        height: 4px;
        background: linear-gradient(90deg, #f96167 0%, #1c7293 100%);
        border-radius: 2px;
        margin-bottom: 2rem;
    }
    .pred-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(202,220,252,0.15);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .pred-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #1c7293;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .pred-class {
        font-family: 'Georgia', serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.1;
        margin: 0.4rem 0;
    }
    .pred-conf {
        font-size: 1.1rem;
        color: #cadcfc;
        margin-top: 0.6rem;
    }
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(202,220,252,0.12);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .stat-num {
        font-family: 'Georgia', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .stat-name {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #1c7293;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin: 0.55rem 0;
    }
    .prob-name {
        flex: 0 0 180px;
        color: #cadcfc;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .prob-bar-bg {
        flex: 1;
        height: 24px;
        background: rgba(255,255,255,0.06);
        border-radius: 6px;
        position: relative;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #1c7293 0%, #065a82 100%);
        border-radius: 6px;
    }
    .prob-pct {
        flex: 0 0 60px;
        text-align: right;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.9rem;
        padding-left: 0.6rem;
    }
    section.main h2, section.main h3 {
        color: #ffffff !important;
        font-family: 'Georgia', serif !important;
    }
    .stMarkdown p {
        color: #cadcfc;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1rem;
        border: 1px dashed rgba(202,220,252,0.25);
    }
    .stSelectbox label, .stCheckbox label, .stSlider label, .stRadio label {
        color: #cadcfc !important;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(202,220,252,0.1);
        color: #6c7a9c;
        font-size: 0.85rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid rgba(202,220,252,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        color: #cadcfc;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 3px solid #f96167 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
@st.cache_resource
def load_model(model_type: str):
    config = load_config()
    device = get_device()
    if model_type == "lstm":
        model = build_lstm(config)
        ckpt_path = Path(config["paths"]["models"]) / "best_lstm_stratified.pt"
    else:
        model = build_transformer(config)
        ckpt_path = Path(config["paths"]["models"]) / "best_transformer_stratified.pt"

    if not ckpt_path.exists():
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, device, config


@st.cache_data
def load_clip_metadata():
    p = Path("data/clips/clip_metadata.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_pose_metadata():
    p = Path("data/poses/pose_metadata.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)


POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 12), (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (24, 26), (25, 27), (26, 28),  # legs
]


def draw_skeleton(frame, landmarks_norm):
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks_norm:
        if np.isnan(lm[0]) or np.isnan(lm[1]):
            pts.append(None)
        else:
            pts.append((int(lm[0] * w), int(lm[1] * h)))
    for i, j in POSE_CONNECTIONS:
        if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
            cv2.line(frame, pts[i], pts[j], (252, 220, 202), 3)
    for p in pts:
        if p is not None:
            cv2.circle(frame, p, 5, (103, 97, 249), -1)
            cv2.circle(frame, p, 5, (255, 255, 255), 1)
    return frame


def render_pred_card(class_name: str, confidence: float):
    st.markdown(
        f"""
        <div class="pred-card">
            <div class="pred-label">PREDICTED STROKE</div>
            <div class="pred-class">{class_name}</div>
            <div class="pred-conf">{confidence*100:.1f}% confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat(label: str, value: str):
    st.markdown(
        f'<div class="stat-card"><div class="stat-num">{value}</div><div class="stat-name">{label}</div></div>',
        unsafe_allow_html=True,
    )


def render_prob_bars(class_names, probs, top_k=5):
    pairs = sorted(zip(class_names, probs), key=lambda x: -x[1])[:top_k]
    html = ""
    for i, (name, p) in enumerate(pairs):
        width_pct = max(p * 100, 1.0)
        grad = "linear-gradient(90deg, #f96167 0%, #d54549 100%)" if i == 0 else "linear-gradient(90deg, #1c7293 0%, #065a82 100%)"
        html += f"""
        <div class="prob-row">
            <div class="prob-name">{name}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: {width_pct}%; background: {grad};"></div>
            </div>
            <div class="prob-pct">{p*100:.1f}%</div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


def compute_detection_rate(keypoints):
    if len(keypoints) == 0:
        return 0.0
    valid_frames = ~np.isnan(keypoints).all(axis=(1, 2))
    return float(valid_frames.mean())


def predict_from_pose(model, device, config, keypoints):
    """Take raw (T, 33, 3) pose → run preprocessing + model → return probs."""
    processed = preprocess_single(keypoints, config, augment=False)
    T, K, D = processed.shape
    flat = processed.reshape(1, T, K * D)
    x = torch.tensor(flat, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


# ============================================================
# UI
# ============================================================
st.markdown('<h1 class="main-title">Badminton-Sense</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">Stroke classification from monocular badminton video — pose-based deep learning</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯  Single Clip", "📊  Match Analytics"])


# ============================================================
# TAB 1 — Single clip prediction
# ============================================================
with tab1:
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])
    with ctrl_col1:
        model_type = st.selectbox("Model", ["lstm", "transformer"], format_func=lambda x: x.upper(), key="single_model")
    with ctrl_col2:
        sample_dir = Path("data/clips")
        sample_clips = sorted(sample_dir.glob("*.mp4"))[:50] if sample_dir.exists() else []
        sample_choice = st.selectbox(
            "Or pick a sample clip",
            ["— upload your own —"] + [p.name for p in sample_clips],
        )
    with ctrl_col3:
        show_skeleton = st.checkbox("Skeleton overlay", value=True)

    uploaded = st.file_uploader("Upload a stroke clip (MP4)", type=["mp4", "avi", "mov"])

    clip_path = None
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(uploaded.read())
        tmp.close()
        clip_path = Path(tmp.name)
    elif sample_choice != "— upload your own —":
        clip_path = sample_dir / sample_choice

    if clip_path is None:
        st.info("👆 Upload a clip or pick a sample to get started.")
    else:
        model, device, config = load_model(model_type)
        class_names = config["classes"]["names"]

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("### Input Clip")
            st.video(str(clip_path))

        with right:
            with st.spinner("Extracting poses..."):
                keypoints, confidences = extract_poses_from_clip(
                    clip_path,
                    model_complexity=config["data"]["mediapipe_model_complexity"],
                )

            detection_rate = compute_detection_rate(keypoints)

            if len(keypoints) < 5 or detection_rate < 0.2:
                st.error(f"Too few valid pose detections (rate: {detection_rate*100:.0f}%). Try another clip.")
            else:
                probs = predict_from_pose(model, device, config, keypoints)
                pred_idx = int(probs.argmax())
                render_pred_card(class_names[pred_idx], float(probs[pred_idx]))

                st.markdown("####")
                s1, s2, s3 = st.columns(3)
                with s1: render_stat("Frames", str(len(keypoints)))
                with s2: render_stat("Detection", f"{detection_rate*100:.0f}%")
                with s3: render_stat("Model", model_type.upper())

                st.markdown("### Top Predictions")
                render_prob_bars(class_names, probs, top_k=5)

                if show_skeleton:
                    st.markdown("### Pose Skeleton")
                    valid_idx = np.where(~np.isnan(keypoints).all(axis=(1, 2)))[0]
                    if len(valid_idx) > 0:
                        default_idx = int(valid_idx[len(valid_idx) // 2])
                        frame_idx = st.slider("Frame", 0, len(keypoints) - 1, default_idx)
                        cap = cv2.VideoCapture(str(clip_path))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            if not np.isnan(keypoints[frame_idx]).all():
                                frame = draw_skeleton(frame.copy(), keypoints[frame_idx])
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)


# ============================================================
# TAB 2 — Match Analytics
# ============================================================
with tab2:
    st.markdown("### Match-Level Stroke Analytics")
    st.markdown(
        '<p style="color:#cadcfc;">Pick an annotated match — model predicts every stroke and aggregates the patterns. Useful for player profiling, training quantification, and tactical analysis.</p>',
        unsafe_allow_html=True,
    )

    clip_meta = load_clip_metadata()
    pose_meta = load_pose_metadata()

    if clip_meta is None or pose_meta is None:
        st.error("Run the data pipeline first (clip + pose extraction).")
    else:
        # Match-name from clip_id (everything before _s1_, _s2_, _s3_)
        def extract_match(cid):
            for tok in ["_s1_", "_s2_", "_s3_"]:
                if tok in cid:
                    return cid.split(tok)[0]
            return cid
        clip_meta = clip_meta.copy()
        clip_meta["match_name"] = clip_meta["clip_id"].apply(extract_match)

        # Merge with pose metadata
        merged = clip_meta.merge(
            pose_meta[["clip_id", "pose_path", "detection_rate", "valid"]],
            on="clip_id", how="left",
        )

        match_options = sorted(merged["match_name"].unique())
        match_pick = st.selectbox("Pick a match", match_options, key="match_pick")
        analytics_model = st.selectbox(
            "Model", ["lstm", "transformer"],
            format_func=lambda x: x.upper(), key="analytics_model",
        )

        match_clips = merged[merged["match_name"] == match_pick].copy()
        match_clips = match_clips[match_clips["valid"] == True].reset_index(drop=True)

        c1, c2, c3 = st.columns(3)
        with c1: render_stat("Total Strokes", str(len(match_clips)))
        with c2:
            n_players = match_clips["player"].nunique() if "player" in match_clips.columns else 0
            render_stat("Players", str(n_players))
        with c3:
            avg_det = match_clips["detection_rate"].mean() * 100 if len(match_clips) else 0
            render_stat("Avg Detection", f"{avg_det:.0f}%")

        if len(match_clips) == 0:
            st.warning("No valid pose sequences for this match.")
        elif st.button("▶  Run model on all strokes", type="primary"):
            model, device, config = load_model(analytics_model)
            class_names = config["classes"]["names"]
            class_mapping = config["classes"]["shuttleset_mapping"]

            preds, true_labels, confidences_list = [], [], []
            progress = st.progress(0)
            status = st.empty()

            for i, row in match_clips.iterrows():
                pose_data = np.load(row["pose_path"], allow_pickle=True).item()
                kps = pose_data["keypoints"]
                try:
                    probs = predict_from_pose(model, device, config, kps)
                    pred_idx = int(probs.argmax())
                    preds.append(pred_idx)
                    confidences_list.append(float(probs[pred_idx]))
                except Exception:
                    preds.append(-1)
                    confidences_list.append(0.0)

                truth = class_mapping.get(row["stroke_type"], "Other")
                true_labels.append(truth)

                if (i + 1) % 20 == 0 or i == len(match_clips) - 1:
                    progress.progress((i + 1) / len(match_clips))
                    status.text(f"Processed {i+1}/{len(match_clips)} strokes")

            status.empty()
            progress.empty()

            # Build results frame
            res = match_clips.copy()
            res["pred_idx"] = preds
            res["pred_class"] = [class_names[p] if p >= 0 else "ERROR" for p in preds]
            res["confidence"] = confidences_list
            res["true_class"] = true_labels

            # ===== Distribution chart =====
            st.markdown("### Predicted Stroke Distribution")
            dist = res["pred_class"].value_counts().reindex(class_names, fill_value=0)
            true_dist = res["true_class"].value_counts().reindex(class_names, fill_value=0)
            chart_df = pd.DataFrame({"Predicted": dist.values, "Annotated (truth)": true_dist.values}, index=class_names)
            st.bar_chart(chart_df, height=320)

            # ===== Per-player breakdown =====
            if "player" in res.columns:
                st.markdown("### Per-Player Stroke Distribution (Predicted)")
                pivot = res.groupby(["player", "pred_class"]).size().unstack(fill_value=0)
                pivot = pivot.reindex(columns=class_names, fill_value=0)
                st.bar_chart(pivot.T, height=320)

            # ===== Accuracy =====
            correct = sum(1 for p, t in zip(res["pred_class"], res["true_class"]) if p == t)
            acc = correct / len(res)
            macro_acc_per_class = {}
            for cn in class_names:
                mask = res["true_class"] == cn
                if mask.sum() > 0:
                    macro_acc_per_class[cn] = (res.loc[mask, "pred_class"] == cn).mean()

            st.markdown("### Model Accuracy on This Match")
            ac1, ac2, ac3 = st.columns(3)
            with ac1: render_stat("Overall Acc", f"{acc*100:.1f}%")
            with ac2: render_stat("Strokes Correct", f"{correct}/{len(res)}")
            with ac3:
                avg_conf = res["confidence"].mean() * 100
                render_stat("Avg Confidence", f"{avg_conf:.0f}%")

            if macro_acc_per_class:
                st.markdown("**Per-class recall (predicted correctly when true label = X):**")
                pc_df = pd.DataFrame.from_dict(macro_acc_per_class, orient="index", columns=["Recall"])
                st.bar_chart(pc_df, height=240)

            # ===== Confidence-filtered samples =====
            st.markdown("### Confident Predictions (showing model's strongest calls)")
            top_conf = res.sort_values("confidence", ascending=False).head(8)
            for _, row in top_conf.iterrows():
                ok = "✅" if row["pred_class"] == row["true_class"] else "❌"
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:0.6rem 1rem; margin:0.3rem 0; color:#cadcfc;">'
                    f'{ok} <strong style="color:#fff;">{row["pred_class"]}</strong> '
                    f'<span style="color:#1c7293;">({row["confidence"]*100:.0f}% conf)</span> · '
                    f'truth: <em>{row["true_class"]}</em> · player {row.get("player", "?")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# Footer
st.markdown(
    '<div class="footer">CS 535 Project — Rutgers — pose extraction via MediaPipe Tasks API · classification via BiLSTM / Spatial-Temporal Transformer</div>',
    unsafe_allow_html=True,
)
