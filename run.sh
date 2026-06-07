#!/usr/bin/env bash
# macOS requires mjpython for the MuJoCo passive viewer.
# Uses Homebrew Python 3.10 mjpython to match venv Python version.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.10/site-packages"

PYTHONPATH="$SITE_PACKAGES" /opt/homebrew/bin/mjpython "$SCRIPT_DIR/main.py" "$@"
