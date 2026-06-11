# SmolVLA Finetuning on AgiBot G2

## 1. Launch EC2

```bash
cd infra
terraform apply   # g6.xlarge spot in us-east-1c (~$0.35/hr)
```

## 2. SSH in and set up

```bash
ssh -i mj3.pem ubuntu@<public_ip>
curl -fsSL https://raw.githubusercontent.com/cactice/ros-study/main/finetune/setup.sh | bash
```

## 3. Download data

```bash
cd ~/ros-study
bash finetune/download_data.sh
```

Downloads a subset of [AgiBot World Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) and converts it to LeRobot format (~10GB).

## 4. Train

```bash
bash finetune/train.sh
```

Checkpoints are saved every 500 steps to `~/checkpoints/smolvla_g2/`.

### Resume after interruption

```bash
RESUME=true bash finetune/train.sh
```

### Sync checkpoints to S3 (recommended for spot instances)

```bash
export S3_BUCKET=s3://your-bucket/smolvla_g2
bash finetune/train.sh
```

The script monitors for spot termination notices and syncs to S3 automatically with a 2-minute window.

### Restore from S3

```bash
aws s3 sync s3://your-bucket/smolvla_g2 ~/checkpoints/smolvla_g2
RESUME=true bash finetune/train.sh
```

## 5. Destroy instance when done

```bash
cd infra
terraform destroy
```
