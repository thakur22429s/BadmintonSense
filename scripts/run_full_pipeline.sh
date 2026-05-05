#!/bin/bash
# Run clip extraction + pose extraction + preprocess + train + eval
# Runs after new videos are downloaded.
set -e

echo "=== Full pipeline run ==="
echo "Start: $(date)"

echo ""
echo "[1/5] Clip extraction (incremental)..."
python -u scripts/03_extract_clips.py 2>&1 | tail -10

echo ""
echo "[2/5] Pose extraction (heavy model, skips existing)..."
python -u scripts/04_extract_poses.py 2>&1 | tail -5

echo ""
echo "[3/5] Preprocess..."
python -u scripts/05_preprocess.py 2>&1 | tail -15

echo ""
echo "[4/5] Train both models..."
python -u scripts/06_train.py --model both --split stratified 2>&1 | grep -E "Training|Stratified|Class weights|Macro-F1|Early stopping|Loaded best"

echo ""
echo "[5/5] Evaluate + figures..."
python -u scripts/07_evaluate.py --model both 2>&1 | grep -E "Evaluating|Macro-F1|Accuracy|All figures"

echo ""
echo "End: $(date)"
echo "=== Pipeline complete ==="
