#!/usr/bin/env bash
# RDT-1B (170M) + MuJoCo G2 viewer
# Physics: rdt-170m on MPS (Apple Silicon GPU via PyTorch)
# Viewer: MuJoCo passive viewer via mjpython
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.10/site-packages"
PYTHONPATH="$SITE_PACKAGES" /opt/homebrew/bin/mjpython "$SCRIPT_DIR/rdt_mujoco.py" "$@"
