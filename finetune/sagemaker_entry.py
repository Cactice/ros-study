"""
SageMaker training entry point.
Runs inside the SageMaker PyTorch container.
Called automatically by SageMaker when the job starts.
"""
import os
import subprocess
import sys

TASK = os.environ.get("TASK", "354")
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
CHECKPOINT_DIR = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    # Install lerobot and dependencies
    run([sys.executable, "-m", "pip", "install", "-q",
         "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@v0.5.1"])

    config_path = os.path.join(os.path.dirname(__file__), f"agibot{TASK}.yaml")

    # Check for existing checkpoint to resume from
    resume = "false"
    if os.path.exists(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
        print(f"[entry] Found existing checkpoint in {CHECKPOINT_DIR}, resuming...")
        resume = "true"

    run([
        "lerobot-train",
        "--config_path", config_path,
        f"--dataset.root={SM_CHANNEL_TRAINING}",
        f"--dataset.repo_id={SM_CHANNEL_TRAINING}",
        f"--output_dir={CHECKPOINT_DIR}",
        f"--resume={resume}",
    ])

    print("[entry] Training complete.", flush=True)


if __name__ == "__main__":
    main()
