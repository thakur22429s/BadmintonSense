#!/bin/bash
# Download 3 matches for pipeline validation (~4,100 strokes)
# Requires: yt-dlp (pip install yt-dlp)

OUTPUT_DIR="data/raw/videos"
mkdir -p "$OUTPUT_DIR"

echo "=== Downloading 3 matches for subset testing ==="

echo "[1/3] Kento MOMOTA vs CHOU Tien Chen - Fuzhou Open 2019 Finals (1,644 strokes)..."
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
  --merge-output-format mp4 \
  -o "$OUTPUT_DIR/Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals.mp4" \
  --no-overwrites \
  "https://www.youtube.com/watch?v=jVGn8EvYxn4" || echo "  FAILED"

echo "[2/3] Anders Antonsen vs Sameer Verma - Thailand Open 2021 QF (1,204 strokes)..."
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
  --merge-output-format mp4 \
  -o "$OUTPUT_DIR/Anders_Antonsen_Sameer_Verma_TOYOTA_THAILAND_OPEN_2021_QuarterFinals.mp4" \
  --no-overwrites \
  "https://www.youtube.com/watch?v=YP8YlZkrQq8" || echo "  FAILED"

echo "[3/3] Ng Ka Long vs Kidambi Srikanth - World Tour Finals 2020 QF (1,200 strokes)..."
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
  --merge-output-format mp4 \
  -o "$OUTPUT_DIR/Ng_Ka_Long_Angus_Kidambi_Srikanth_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals.mp4" \
  --no-overwrites \
  "https://www.youtube.com/watch?v=nDBhJhG0fVU" || echo "  FAILED"

echo ""
echo "=== Done. Check $OUTPUT_DIR for downloaded files ==="
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "No files found."
