#!/bin/bash
# Download 4 additional matches to scale dataset from 3 → 7 matches
# Run: bash scripts/02_download_videos_more.sh

OUTPUT_DIR="data/raw/videos"
mkdir -p "$OUTPUT_DIR"

echo "=== Downloading 4 additional matches ==="

declare -a MATCHES=(
  "Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals|https://www.youtube.com/watch?v=5W6txLGZ1Rs"
  "Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final|https://www.youtube.com/watch?v=yu9oyMXRGHY"
  "Kento_MOMOTA_Viktor_AXELSEN_Malaysia_Masters_2020_Finals|https://www.youtube.com/watch?v=boQC4J4E1ZQ"
  "CHOU_Tien_Chen_Anders_ANTONSEN_Fuzhou_Open_2019_Semi-finals|https://www.youtube.com/watch?v=32j2Tg64Zbg"
)

i=1
total=${#MATCHES[@]}
for entry in "${MATCHES[@]}"; do
  IFS='|' read -r name url <<< "$entry"
  echo ""
  echo "[$i/$total] $name"
  if [ -f "$OUTPUT_DIR/${name}.mp4" ]; then
    echo "  Already downloaded, skipping."
  else
    python -m yt_dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
      --merge-output-format mp4 \
      -o "$OUTPUT_DIR/${name}.mp4" \
      --no-overwrites \
      "$url" 2>&1 | tail -3 || echo "  FAILED"
  fi
  i=$((i+1))
done

echo ""
echo "=== Downloads complete ==="
ls -lh "$OUTPUT_DIR"/*.mp4
