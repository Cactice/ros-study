output "ssh_command" {
  description = "SSH into the instance"
  value       = "ssh -i ../mj3.pem ubuntu@${aws_spot_instance_request.trainer.public_ip}"
}

output "public_ip" {
  value = aws_spot_instance_request.trainer.public_ip
}
