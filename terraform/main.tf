terraform {
  required_version = ">= 1.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    bucket         = "flash-loan-bot-terraform-state"
    key            = "global/s3/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "flash-loan-bot-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = "flash-loan-bot"
      ManagedBy   = "terraform"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Remote state for shared resources
data "terraform_remote_state" "shared" {
  backend = "s3"
  config = {
    bucket = "flash-loan-bot-terraform-state"
    key    = "shared/terraform.tfstate"
    region = "us-east-1"
  }
} 