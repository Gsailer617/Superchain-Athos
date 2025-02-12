variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (e.g., prod, staging)"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.large"
}

variable "key_name" {
  description = "Name of SSH key pair"
  type        = string
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
}

variable "monitoring" {
  description = "Monitoring configuration"
  type = object({
    grafana_admin_password = string
    alertmanager_slack_url = string
    prometheus_retention   = string
  })
  default = {
    grafana_admin_password = "admin"
    alertmanager_slack_url = ""
    prometheus_retention   = "15d"
  }
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in the cluster"
  type        = number
  default     = 2
} 