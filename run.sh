#!/usr/bin/env bash
# MuJoCo viewer (plain physics step via MJX)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.10/site-packages"
PYTHONPATH="$SITE_PACKAGES" /opt/homebrew/bin/mjpython "$SCRIPT_DIR/main.py" "$@"
