#!/bin/bash
# Run once on a fresh EC2 instance to install dependencies.
set -e

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Clone repo
git clone https://github.com/cactice/ros-study.git ~/ros-study
cd ~/ros-study

# Install Python deps
uv sync

# Install any4lerobot (AgiBot → LeRobot format converter)
pip install -q git+https://github.com/Tavish9/any4lerobot.git

# Install AWS CLI (for S3 checkpoint sync)
pip install -q awscli

echo "Setup complete. Next: run download_data.sh, then train.sh"
