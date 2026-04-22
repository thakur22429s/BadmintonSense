# Badminton-Sense: Project Progress Summary

**Course:** CS 535 — Rutgers University
**Participant:** Abhay Singh Thakur (Individual Project)
**Date:** April 21, 2026

---

## 1. Project Title & Objective

**Title:** Badminton-Sense: Deep Learning Stroke Classification from Monocular Badminton Video via Spatio-Temporal Pose Estimation

**Major Objective:** Build an end-to-end pipeline that takes raw BWF (Badminton World Federation) broadcast match video and classifies individual stroke types (serve, smash, drop, clear, etc.) using 2D pose estimation as an intermediate representation, comparing a BiLSTM baseline against a Spatial-Temporal Transformer architecture.

**Pipeline:**
```
BWF Match Video → Clip Stroke Segments → MediaPipe Pose Extraction → Hip-Center Normalization → LSTM/Transformer → Stroke Class (10 types)
```

---

## 2. Dataset

- **Source:** ShuttleSet (KDD 2023) from the CoachAI-Projects GitHub repo
- **Full dataset:** 36,492 annotated strokes across 44 BWF professional matches
- **Annotations:** Chinese-language stroke labels mapped to a 10-class English taxonomy
- **Subset used for validation:** 3 matches (4,048 strokes → 1,721 valid pose sequences after filtering)

### 10-Class Stroke Taxonomy (mapped from 19 ShuttleSet Chinese labels):
| Class | Description | ShuttleSet Labels |
|-------|-------------|-------------------|
| Serve | Short/long serve | 發短球, 發長球 |
| Clear/Lob | Overhead clears and lobs | 長球, 挑球 |
| Drop | Slice/drop shots | 切球, 過度切球 |
| Smash | Full and half smashes | 殺球, 點扣 |
| Drive | Flat drives | 平球, 後場抽平球, 小平球 |
| Net Shot | Net drops and cross-net | 放小球, 勾球 |
| Push/Rush | Net pushes and kills | 推球, 撲球 |
| Defensive Return | Blocks and defensive lifts | 擋小球, 防守回抽, 防守回挑 |
| Cross-court | (Empty in subset — 0 samples) | — |
| Other | Unknown stroke types | 未知球種 |

### Class Distribution in Subset (1,721 valid sequences):
| Class | Count | % |
|-------|-------|---|
| Clear/Lob | 414 | 24.1% |
| Defensive Return | 272 | 15.8% |
| Net Shot | 261 | 15.2% |
| Smash | 203 | 11.8% |
| Drop | 169 | 9.8% |
| Push/Rush | 159 | 9.2% |
| Serve | 110 | 6.4% |
| Other | 91 | 5.3% |
| Drive | 42 | 2.4% |
| Cross-court | 0 | 0.0% |

---

## 3. Architecture Details

### Model A: BiLSTM Baseline
- 2-layer Bidirectional LSTM (hidden_dim=128)
- Input: flattened keypoint sequences (batch, 30 frames, 45 features = 15 keypoints × 3 coords)
- FC head: 256-dim → 10 classes
- Parameters: ~527K
- Optimizer: Adam, LR=1e-3, ReduceLROnPlateau scheduler

### Model B: Spatial-Temporal Transformer (BST-inspired)
- Spatial attention block: embed_dim=64, 2 heads, 2 layers (attends across joints within each frame)
- Temporal attention block: embed_dim=128, 4 heads, 3 layers (attends across frames)
- Per-keypoint embedding: 32-dim
- Parameters: ~841K
- Optimizer: AdamW, LR=5e-4, weight_decay=1e-4, Cosine Annealing scheduler with 5 warmup epochs

### Shared Training Config:
- Label smoothing cross-entropy (ε=0.1)
- Gradient clipping (max_norm=1.0)
- Early stopping patience: 10 epochs
- Max epochs: 100
- Data split: 70/15/15 stratified (train=1,204, val=258, test=259)

---

## 4. Pose Extraction Pipeline

- **Tool:** MediaPipe PoseLandmarker (Tasks API, v0.10.33)
- **Model:** Lite model (5.5 MB) for subset validation, Heavy model (29.2 MB) available for full run
- **Running mode:** IMAGE mode (processes each frame independently — allows model reuse across clips)
- **Selected keypoints:** 15 of 33 MediaPipe landmarks (nose, shoulders, elbows, wrists, hands, hips, knees, ankles)
- **Coordinate dimensions:** 3 (x, y, z)
- **Normalization:** Hip-center translation + torso-length scaling for size/position invariance
- **Temporal resampling:** All clips resampled to fixed 30 frames
- **Minimum detection rate:** 0.5 (50% of frames must have valid pose detection)
- **Result:** 1,721 of 4,048 clips passed the detection threshold (42.5% valid rate)

---

## 5. Results (3-Match Subset Validation)

### Test Set Performance

| Metric | BiLSTM | Transformer |
|--------|--------|-------------|
| **Macro-F1** | **0.139** | 0.085 |
| **Weighted-F1** | **0.177** | 0.115 |
| **Accuracy** | 22.4% | 25.1% |
| Best Epoch | 33 (of 43, early stopped) | 25 (of 35, early stopped) |

### Per-Class F1 Scores (Test Set)

| Class | BiLSTM | Transformer |
|-------|--------|-------------|
| Serve | 0.00 | 0.00 |
| Clear/Lob | 0.32 | 0.38 |
| Drop | 0.00 | 0.00 |
| Smash | 0.08 | 0.00 |
| Drive | 0.00 | 0.00 |
| Net Shot | 0.17 | 0.00 |
| Push/Rush | 0.00 | 0.00 |
| Defensive Return | 0.22 | 0.00 |
| Cross-court | 0.00 | 0.00 |
| Other | 0.59 | 0.47 |

### Key Findings:
1. **BiLSTM outperforms Transformer** on this small subset — learned partial signal for 5 of 10 classes vs Transformer's 2
2. **Transformer collapsed to majority-class prediction** — predicts nearly everything as Clear/Lob (94%+ of predictions). Classic behavior when transformer architectures are data-starved (only 1,204 training samples)
3. **"Other" class surprisingly well-classified** by both models — likely has distinctive pose patterns (uncertain/unusual movements)
4. **Clear/Lob is the majority class** (24%) and dominates both models' predictions
5. **Training curves show LSTM overfitting** — train loss keeps dropping while val loss diverges after epoch ~15, suggesting more data or regularization would help
6. **t-SNE embeddings** show LSTM learned more structured representations (visible clusters) while Transformer embeddings are more diffuse/uniform

### Generated Evaluation Artifacts:
- `results/figures/confusion_matrix_lstm.png` — Normalized confusion matrix
- `results/figures/confusion_matrix_transformer.png`
- `results/figures/training_curves_lstm.png` — Loss and F1 over epochs
- `results/figures/training_curves_transformer.png`
- `results/figures/per_class_f1_lstm.png` — Bar chart of per-class F1
- `results/figures/per_class_f1_transformer.png`
- `results/figures/tsne_lstm.png` — t-SNE of learned embeddings
- `results/figures/tsne_transformer.png`

---

## 6. Milestones Accomplished

### ✅ Design & Specification
- Defined 10-class stroke taxonomy mapped from ShuttleSet's 19 Chinese stroke labels
- Designed full 7-step pipeline (download → annotate → clip → pose → preprocess → train → evaluate)
- Specified both model architectures (BiLSTM baseline + Spatial-Temporal Transformer)

### ✅ Implementation
- Complete end-to-end pipeline: 7 Python scripts, modular `src/` package
- Adapted to MediaPipe Tasks API (v0.10.33) — the legacy `mp.solutions.pose` API was removed
- Hip-center normalization and torso-length scaling for pose invariance
- Label smoothing cross-entropy loss
- Stratified splitting with fallback for small match counts
- Full evaluation suite (confusion matrices, training curves, per-class F1, t-SNE)
- Streamlit demo app scaffold

### ✅ Testing & Validation
- Validated full pipeline end-to-end on 3-match subset (1,721 sequences)
- Both models train, converge, and produce meaningful (if modest) predictions
- Generated all evaluation visualizations
- Confirmed pipeline scales — processing time dominated by pose extraction (~2 hrs for 3 matches on CPU with lite model)

### 🔲 Remaining Work
- Scale to full 44 matches (36,492 strokes) — expect significantly better results
- Use heavy pose model for better keypoint quality
- GPU training for faster iteration and hyperparameter search
- Data augmentation (temporal jitter, Gaussian noise, horizontal flip, temporal scaling, frame dropout — all implemented but need tuning)
- Complete Streamlit demo app for live inference
- Final write-up and analysis

---

## 7. Changes from Original Proposal

### Change 1: MediaPipe API Migration
- **Original plan:** Use `mp.solutions.pose.Pose()` (legacy MediaPipe API)
- **What changed:** MediaPipe v0.10.33 completely removed the legacy `mp.solutions` API
- **New approach:** Full rewrite to use the `PoseLandmarker` Tasks API with `.task` model files
- **Impact:** Required downloading separate model weight files (5.5-29.2 MB) and learning the new API surface. Used IMAGE mode instead of VIDEO mode to allow model reuse across clips.

### Change 2: Subset Validation Before Full Run
- **Original plan:** Train on the full 44-match ShuttleSet dataset
- **What changed:** Running the full pipeline on 44 matches would take 30+ hours on CPU (video download + pose extraction). Validated on 3 matches first.
- **Reason:** De-risk the pipeline — ensure every step works before committing to the full data processing run
- **Impact:** Results are modest (expected) but the pipeline is proven end-to-end

### Change 3: Lowered Pose Detection Threshold
- **Original plan:** Require 70% of frames to have valid pose detection (min_detection_rate=0.7)
- **What changed:** Lowered to 50% (min_detection_rate=0.5)
- **Reason:** Broadcast badminton video has distant wide-angle cameras where players are small in frame — only 32% of clips passed at the 0.7 threshold. At 0.5, we retain 42.5% (1,721 of 4,048 clips).
- **Impact:** More training data at the cost of noisier pose sequences

### Change 4: Sample-Level Splitting Instead of Match-Level
- **Original plan:** Leave-one-match-out (LOSO) or match-level stratified splitting
- **What changed:** With only 3 matches, match-level splits produced empty validation/test sets. Fell back to sample-level stratified splitting.
- **Reason:** int(3 × 0.15) = 0 matches for val/test
- **Impact:** Temporary for subset validation only. Full 44-match run will use match-level LOSO as originally planned.

### Change 5: Lite Pose Model for Validation
- **Original plan:** Use MediaPipe Heavy model (model_complexity=2) for best pose quality
- **What changed:** Used Lite model (model_complexity=0) for the 3-match subset
- **Reason:** Processing speed — lite model is ~3x faster, and the goal was pipeline validation, not final results
- **Impact:** Lower pose quality but adequate for proving the pipeline works. Full run will use the heavy model.

---

## 8. Technical Environment
- Python 3.14.2, PyTorch 2.11.0+cpu, mediapipe 0.10.33, scikit-learn 1.8.0
- Windows 11, CPU-only training (no CUDA GPU available locally)
- yt-dlp for video download, ffmpeg for clip extraction
- Tools: matplotlib, seaborn for visualization; tqdm for progress; numpy for numerics

---

## 9. Expected Results & Conclusions (Full Dataset)

### Expected Results:
- With 44 matches (~15,000-25,000 valid pose sequences after filtering), expect macro-F1 in the 0.35-0.55 range for BiLSTM and 0.30-0.50 for Transformer
- Heavy pose model should improve detection rate from 42.5% to 55-65%
- GPU training would enable proper hyperparameter tuning and data augmentation

### Expected Conclusions:
1. Pose-based stroke classification from broadcast video is feasible but challenging due to distant camera angles
2. BiLSTM is a strong baseline for temporal pose classification when data is limited
3. Spatial-Temporal Transformers need substantially more data to outperform LSTMs on this task
4. MediaPipe pose estimation on broadcast badminton video has significant limitations (low detection rate, noisy landmarks) — purpose-trained pose models may be needed for production use
5. The 10-class taxonomy captures meaningful stroke distinctions, though some classes (Serve, Drop, Push/Rush) may be harder to distinguish from pose alone without ball trajectory information
