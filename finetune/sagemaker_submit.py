"""
Submit a SageMaker on-demand training job.

Usage:
    python finetune/sagemaker_submit.py            # default 16hr cap
    MAX_HOURS=8 python finetune/sagemaker_submit.py

Job stops automatically after MAX_HOURS. Checkpoints are saved to S3
every few minutes so progress is never fully lost.
"""
import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

BUCKET = "smolvla-checkpoints-206078779659"
ROLE_NAME = "smolvla-sagemaker-role"
INSTANCE_TYPE = "ml.g6.xlarge"   # L4 GPU, 24GB VRAM, $1.127/hr on-demand
TASK = "354"
MAX_HOURS = int(os.environ.get("MAX_HOURS", 16))


def main() -> None:
    session = sagemaker.Session()
    account = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account}:role/{ROLE_NAME}"

    estimated_cost = MAX_HOURS * 1.127
    print(f"Instance:   {INSTANCE_TYPE} @ $1.127/hr on-demand")
    print(f"Max run:    {MAX_HOURS} hrs (hard cap — job killed after this)")
    print(f"Max cost:   ~${estimated_cost:.2f}")

    estimator = PyTorch(
        entry_point="sagemaker_entry.py",
        source_dir="finetune/",
        role=role_arn,
        framework_version="2.1",
        py_version="py310",
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        use_spot_instances=False,
        max_run=MAX_HOURS * 3600,  # hard stop — SageMaker kills the job
        checkpoint_s3_uri=f"s3://{BUCKET}/checkpoints/task{TASK}/",
        checkpoint_local_path="/opt/ml/checkpoints",
        environment={"TASK": TASK},
        sagemaker_session=session,
    )

    data_uri = f"s3://{BUCKET}/dataset/task{TASK}/"
    print(f"\nSubmitting job — data: {data_uri}")
    print(f"Checkpoints: s3://{BUCKET}/checkpoints/task{TASK}/")

    estimator.fit({"training": data_uri}, wait=False)
    print(f"\nJob submitted: {estimator.latest_training_job.name}")
    print("Monitor: https://console.aws.amazon.com/sagemaker/home#/jobs")


if __name__ == "__main__":
    main()
