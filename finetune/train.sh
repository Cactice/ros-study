#!/bin/bash
set -e

source "$HOME/.local/bin/env" 2>/dev/null || true
source "$HOME/ros-study/.venv/bin/activate" 2>/dev/null || true

TASK="${TASK:-354}"
S3_BUCKET="${S3_BUCKET:-s3://smolvla-checkpoints-206078779659}"
OUTPUT_DIR="$HOME/checkpoints/smolvla_g2_task${TASK}"
RESUME="${RESUME:-false}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Background loop: sync to S3 every 5 minutes AND on spot termination warning
(
  while true; do
    sleep 300
    aws s3 sync "$OUTPUT_DIR" "$S3_BUCKET/task${TASK}/" --quiet 2>/dev/null || true
    # Check for spot termination (2-min warning)
    if curl -sf --max-time 2 http://169.254.169.254/latest/meta-data/spot/termination-time > /dev/null 2>&1; then
      echo "[sync] Spot termination detected — final sync..."
      aws s3 sync "$OUTPUT_DIR" "$S3_BUCKET/task${TASK}/" --quiet 2>/dev/null || true
      break
    fi
  done
) &
SYNC_PID=$!

echo "[train] Output: $OUTPUT_DIR"
echo "[train] S3: $S3_BUCKET/task${TASK}/"
echo "[train] Sync PID: $SYNC_PID"

lerobot-train \
  --config_path "$SCRIPT_DIR/agibot${TASK}.yaml" \
  --output_dir "$OUTPUT_DIR" \
  --resume "$RESUME"

# Final sync after training completes
echo "[train] Training done. Final sync to S3..."
aws s3 sync "$OUTPUT_DIR" "$S3_BUCKET/task${TASK}/" --quiet
echo "[train] Done."

kill $SYNC_PID 2>/dev/null || true
