#!/usr/bin/env bash
# SmolVLA (450M) + MuJoCo G2 viewer
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.10/site-packages"
PYTHONPATH="$SITE_PACKAGES" /opt/homebrew/bin/mjpython "$SCRIPT_DIR/smolvla_mujoco.py" "$@"
