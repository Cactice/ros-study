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
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'source $HOME/.local/bin/env' >> /home/ubuntu/.bashrc
    # Install HuggingFace CLI
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

