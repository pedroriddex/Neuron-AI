terraform {
  required_version = ">= 1.2"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "project_prefix" {
  type        = string
  description = "Prefix for resource names (e.g., neuron-mvp)"
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}
