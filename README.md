# Badminton-Sense

Deep learning stroke classification from monocular badminton video via spatio-temporal pose estimation.

**CS 535 Individual Project — Rutgers University**

## Pipeline

```
BWF Match Video → Clip Stroke Segments → MediaPipe Pose → Normalize → LSTM/Transformer → Stroke Class
```

## Setup

```bash
pip install -r requirements.txt
```

Requires: Python 3.10+, ffmpeg, yt-dlp

## Usage

Run scripts in order:

```bash
python scripts/01_download_data.py      # Download ShuttleSet annotations
bash scripts/02_download_videos.sh       # Download BWF match videos (yt-dlp)
python scripts/03_extract_clips.py       # Segment into stroke clips
python scripts/04_extract_poses.py       # MediaPipe pose extraction
python scripts/05_preprocess.py          # Normalize + split
python scripts/06_train.py --model both  # Train LSTM + Transformer
python scripts/07_evaluate.py            # Generate evaluation plots
```

## Demo

### Live Vercel showcase (Next.js)

> Live demo URL: _paste Vercel URL here after deploy_

To deploy:

1. Go to https://vercel.com/new
2. Import `thakur22429s/BadmintonSense`
3. Set Root Directory: `web`
4. Framework Preset: Next.js (auto-detected)
5. Build Command: `npm run build` (default)
6. Output Directory: `.next` (default)
7. Click Deploy
8. After build completes, paste the URL above

### Streamlit (legacy demo)

```bash
streamlit run app/streamlit_app.py
```

## Models

- **BiLSTM**: 2-layer bidirectional LSTM on flattened keypoint sequences
- **Spatial-Temporal Transformer**: BST-inspired architecture with separate spatial (inter-joint) and temporal attention

## Dataset

Built on [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects) (KDD 2023) — 36,492 strokes across 44 BWF matches.

## Tests

```bash
pytest tests/ -v
```
