#!/bin/bash
set -e

OUTPUT_DIR="$HOME/checkpoints/smolvla_g2"
RESUME="${RESUME:-false}"

# Sync checkpoint to S3 on spot termination (2-min warning)
if [ -n "$S3_BUCKET" ]; then
  ( while true; do
      curl -sf --max-time 2 http://169.254.169.254/latest/meta-data/spot/termination-time \
        && aws s3 sync "$OUTPUT_DIR" "$S3_BUCKET" --quiet && break
      sleep 5
    done ) &
fi

TASK="${TASK:-354}"

lerobot-train \
  --policy.type smolvla \
  --policy.pretrained lerobot/smolvla_base \
  --dataset.repo_id "$HOME/data/agibot_lerobot/task_${TASK}" \
  --output_dir "$OUTPUT_DIR" \
  --resume "$RESUME" \
  --save_checkpoint true \
  --save_freq 500 \
  --training.num_steps 10000 \
  --training.batch_size 16 \
  --device cuda

[ -n "$S3_BUCKET" ] && aws s3 sync "$OUTPUT_DIR" "$S3_BUCKET" --quiet
