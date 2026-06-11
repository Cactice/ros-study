#!/bin/bash
# Download AgiBot World Alpha in pre-converted LeRobot format.
# Source: Traly/AgiBotWorld-Alpha-lerobot (CC BY-NC-SA 4.0, non-commercial use only)
set -e

LEROBOT_DIR="$HOME/data/agibot_lerobot"
# Task IDs to download (default: 354 = "Pickup items in supermarket", 2 steps, 516 episodes ~21GB)
TASKS="${TASKS:-354}"

mkdir -p "$LEROBOT_DIR"

echo "Downloading AgiBot World Alpha (tasks: $TASKS) in LeRobot format..."
for task_id in $TASKS; do
  echo "  task_$task_id..."
  hf download Traly/AgiBotWorld-Alpha-lerobot \
    --repo-type dataset \
    --local-dir "$LEROBOT_DIR" \
    --include "task_${task_id}/**"
done

echo "Done. Dataset at: $LEROBOT_DIR"
du -sh "$LEROBOT_DIR"
