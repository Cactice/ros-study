#!/usr/bin/env bash
# macOS requires mjpython for the MuJoCo passive viewer.
# Physics is computed via MJX (JAX) on Metal GPU.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.12/site-packages"
MJPYTHON="$(python3 -c "import mujoco, os; print(os.path.join(os.path.dirname(mujoco.__file__), 'MuJoCo_(mjpython).app', 'Contents', 'MacOS', 'mjpython'))" 2>/dev/null || echo "/opt/homebrew/bin/mjpython")"

PYTHONPATH="$SITE_PACKAGES" "$MJPYTHON" "$SCRIPT_DIR/main.py" "$@"
