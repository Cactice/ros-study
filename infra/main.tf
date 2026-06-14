terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# IAM role so the instance can write checkpoints to S3
resource "aws_iam_role" "trainer" {
  name = "smolvla-trainer-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "trainer_s3" {
  name = "smolvla-trainer-s3"
  role = aws_iam_role.trainer.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket", "s3:DeleteObject"]
      Resource = [
        "arn:aws:s3:::${var.checkpoint_bucket}",
        "arn:aws:s3:::${var.checkpoint_bucket}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "trainer" {
  name = "smolvla-trainer-profile"
  role = aws_iam_role.trainer.name
}

resource "aws_security_group" "smolvla" {
  name        = "smolvla-training"
  description = "SSH access for SmolVLA training"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_spot_instance_request" "trainer" {
  ami                            = data.aws_ami.dlami.id
  instance_type                  = var.instance_type
  key_name                       = var.key_name
  security_groups                = [aws_security_group.smolvla.name]
  iam_instance_profile           = aws_iam_instance_profile.trainer.name
  availability_zone              = "us-east-1c"
  spot_price                     = var.spot_price
  wait_for_fulfillment           = true
  instance_interruption_behavior = "terminate"

  root_block_device {
    volume_type = "gp3"
    volume_size = var.disk_gb
    iops        = 3000
    throughput  = 125
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'source $HOME/.local/bin/env' >> /home/ubuntu/.bashrc
    pip install -q huggingface_hub
    echo "Instance ready."
  EOF

  tags = {
    Name = "smolvla-trainer"
  }
}

output "instance_id" {
  value = aws_spot_instance_request.trainer.spot_instance_id
}

# ── SageMaker execution role ──────────────────────────────────────────────────

resource "aws_iam_role" "sagemaker" {
  name = "smolvla-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "smolvla-sagemaker-s3"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${var.checkpoint_bucket}",
        "arn:aws:s3:::${var.checkpoint_bucket}/*",
      ]
    }]
  })
}

