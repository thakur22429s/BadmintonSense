"""Streamlit demo: upload a badminton clip and get stroke classification."""

import sys
sys.path.insert(0, ".")

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from src.data.download import load_config
from src.data.pose_extractor import extract_poses_from_clip
from src.data.preprocessing import preprocess_single
from src.models.lstm import build_lstm
from src.models.transformer import build_transformer
from src.models.utils import get_device


@st.cache_resource
def load_model(config, model_type="transformer"):
    device = get_device()
    if model_type == "lstm":
        model = build_lstm(config)
        checkpoint_path = Path(config["paths"]["models"]) / "best_lstm_stratified.pt"
    else:
        model = build_transformer(config)
        checkpoint_path = Path(config["paths"]["models"]) / "best_transformer_stratified.pt"

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def draw_skeleton(frame, landmarks, connections=None):
    """Draw pose skeleton on a video frame."""
    h, w = frame.shape[:2]
    for lm in landmarks:
        x, y = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    if connections:
        for i, j in connections:
            if i < len(landmarks) and j < len(landmarks):
                pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
                pt2 = (int(landmarks[j][0] * w), int(landmarks[j][1] * h))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
    return frame


POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 12), (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (24, 26),  # legs
]


def main():
    st.set_page_config(page_title="Badminton-Sense", layout="wide")
    st.title("Badminton-Sense: Stroke Classification")
    st.markdown("Upload a badminton stroke clip to classify the stroke type using pose estimation.")

    config = load_config()
    class_names = config["classes"]["names"]

    # Sidebar
    st.sidebar.header("Settings")
    model_type = st.sidebar.selectbox("Model", ["transformer", "lstm"])
    show_skeleton = st.sidebar.checkbox("Show Skeleton Overlay", value=True)

    # Load model
    model, device = load_model(config, model_type)

    # File upload
    uploaded = st.file_uploader("Upload a stroke clip (MP4)", type=["mp4", "avi", "mov"])

    # Sample clips
    sample_dir = Path(config["demo"]["sample_clips_dir"])
    sample_clips = list(sample_dir.glob("*.mp4")) if sample_dir.exists() else []
    if sample_clips:
        st.markdown("**Or choose a sample clip:**")
        selected_sample = st.selectbox("Sample clips", ["None"] + [p.name for p in sample_clips])
        if selected_sample != "None":
            uploaded = sample_dir / selected_sample

    if uploaded is not None:
        # Save uploaded file
        if hasattr(uploaded, "read"):
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(uploaded.read())
            clip_path = Path(tmp.name)
        else:
            clip_path = Path(uploaded)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Video")
            st.video(str(clip_path))

        # Extract poses
        with st.spinner("Extracting poses..."):
            keypoints, confidences = extract_poses_from_clip(clip_path)

        detection_rate = np.any(~np.isnan(keypoints).all(axis=(1, 2))).mean() if len(keypoints) > 0 else 0
        st.metric("Detection Rate", f"{detection_rate*100:.1f}%")

        if len(keypoints) < 5:
            st.error("Too few frames with valid pose detection. Try a different clip.")
            return

        # Preprocess
        processed = preprocess_single(keypoints, config, augment=False)
        T, K, D = processed.shape
        flat = processed.reshape(1, T, K * D)
        input_tensor = torch.tensor(flat, dtype=torch.float32).to(device)

        # Predict
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_class = class_names[probs.argmax()]
        confidence = probs.max()

        with col2:
            st.subheader("Prediction")
            st.metric("Stroke Type", predicted_class)
            st.metric("Confidence", f"{confidence*100:.1f}%")

            st.markdown("**Class Probabilities:**")
            for name, prob in sorted(zip(class_names, probs), key=lambda x: -x[1]):
                st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

        # Skeleton overlay
        if show_skeleton and len(keypoints) > 0:
            st.subheader("Pose Visualization")
            frame_idx = st.slider("Frame", 0, len(keypoints) - 1, len(keypoints) // 2)

            cap = cv2.VideoCapture(str(clip_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret and not np.isnan(keypoints[frame_idx]).all():
                frame_with_skeleton = draw_skeleton(
                    frame.copy(), keypoints[frame_idx], POSE_CONNECTIONS
                )
                st.image(cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB))

        # Trajectory visualization
        st.subheader("Keypoint Trajectories")
        selected_kp = config["data"]["selected_keypoints"]
        kp_names = ["nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
                    "L_wrist", "R_wrist", "L_pinky", "R_pinky", "L_index",
                    "R_index", "L_hip", "R_hip", "L_knee", "R_knee"]
        chosen_kp = st.selectbox("Keypoint", kp_names, index=6)  # R_wrist default
        kp_idx = kp_names.index(chosen_kp)
        mp_idx = selected_kp[kp_idx]

        if mp_idx < keypoints.shape[1]:
            trajectory = keypoints[:, mp_idx, :2]
            valid = ~np.isnan(trajectory).any(axis=1)
            if valid.any():
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(trajectory[valid, 0], trajectory[valid, 1], "b-", alpha=0.7)
                ax.scatter(trajectory[valid, 0][0], trajectory[valid, 1][0], c="green", s=50, label="Start")
                ax.scatter(trajectory[valid, 0][-1], trajectory[valid, 1][-1], c="red", s=50, label="End")
                ax.set_title(f"{chosen_kp} Trajectory")
                ax.legend()
                ax.invert_yaxis()
                st.pyplot(fig)


if __name__ == "__main__":
    main()
