variable "region" {
  default = "us-east-1"
}

variable "instance_type" {
  default = "g6.xlarge"
}

variable "key_name" {
  description = "EC2 key pair name (without .pem)"
  default     = "mj3"
}

variable "disk_gb" {
  default = 100
}

variable "spot_price" {
  description = "Max spot price in USD/hr (on-demand is ~$1.01)"
  default     = "0.50"
}
