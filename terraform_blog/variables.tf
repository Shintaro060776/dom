variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  default     = "192.168.0.0/16"
}

variable "subnet_cidr" {
  description = "List of CIDR blocks"
  type        = list(string)
  default     = ["192.168.1.0/24", "192.168.2.0/24"]
}

variable "vpc_name" {
  description = "name of vpc"
  default     = "blog"
}

variable "availability_zone1" {
  description = "availability zone 1"
  default     = "ap-northeast-1a"
}

variable "availability_zone2" {
  description = "availability zone 2"
  default     = "ap-northeast-1c"
}

variable "subnet_names" {
  description = "a list of names for subnets"
  type        = list(string)
  default     = ["blog", "blog2"]
}

variable "internet_gateway_name" {
  description = "name of internet gateway"
  default     = "blog"
}

variable "aws_route_table" {
  description = "name of route table"
  default     = "blog"
}

variable "cidr_block" {
  description = "default route"
  default     = "0.0.0.0/0"
}

variable "aws_lb_name" {
  description = "Name of the AWS Load Balancer."
  default     = "blog-lb"
}

variable "aws_lb_listener_port" {
  description = "Port for the AWS Load Balancer listener."
  default     = 80
}

variable "aws_lb_listener_protocol" {
  description = "Protocol for the AWS Load Balancer listener."
  default     = "HTTP"
}

variable "aws_lb_target_group_name" {
  description = "Name for the AWS Load Balancer target group."
  default     = "blog-tg"
}

variable "aws_lb_target_group_protocol" {
  description = "Protocol for the AWS Load Balancer target group."
  default     = "HTTP"
}

variable "ecs_cluster_name" {
  description = "Name of the ECS cluster."
  default     = "blog-cluster"
}

variable "ecs_service_name" {
  description = "Name of the ECS service."
  default     = "blog-service"
}

variable "ecs_task_family" {
  description = "Family name of the ECS task definition."
  default     = "blog-task"
}

variable "ecs_task_cpu" {
  description = "CPU allocation for the ECS task."
  default     = "256"
}

variable "ecs_task_memory" {
  description = "Memory allocation for the ECS task."
  default     = "512"
}

variable "ecs_container_name" {
  description = "Name of the container in the ECS task."
  default     = "blog-container"
}

variable "ecs_container_image" {
  description = "Image of the container in the ECS task."
  default     = "715573459931.dkr.ecr.ap-northeast-1.amazonaws.com/blog:latest"
}

variable "ecs_container_port" {
  description = "Port of the container in the ECS task."
  default     = 80
}

variable "ecs_execution_role_name" {
  description = "Name of the ECS execution role."
  default     = "ecs_execution_role"
}

variable "eks_cluster_name" {
  description = "Name of the EKS cluster."
  default     = "blog"
}

variable "eks_cluster_role_name" {
  description = "Name of the IAM role for EKS cluster."
  default     = "blog-eks-cluster-role"
}

variable "eks_node_role_name" {
  description = "Name of the IAM role for EKS nodes."
  default     = "blog-eks-node-role"
}

variable "eks_security_group_name" {
  description = "Name of the security group for the EKS cluster."
  default     = "blog"
}

variable "eks_node_group_name" {
  description = "Name of the EKS node group."
  default     = "blog"
}

variable "eks_node_group_desired_size" {
  description = "Desired size of the EKS node group."
  default     = 1
}

variable "eks_node_group_max_size" {
  description = "Maximum size of the EKS node group."
  default     = 2
}

variable "eks_node_group_min_size" {
  description = "Minimum size of the EKS node group."
  default     = 1
}