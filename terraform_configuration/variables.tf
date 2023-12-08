variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  default     = "172.16.0.0/16"
}

variable "subnet_cidr" {
  description = "List of CIDR blocks"
  type        = list(string)
  default     = ["172.16.1.0/24", "172.16.2.0/24"]
}

variable "vpc_name" {
  description = "name of vpc"
  default     = "dom"
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
  default     = ["dom", "dom2"]
}

variable "internet_gateway_name" {
  description = "name of internet gateway"
  default     = "dom"
}

variable "aws_route_table" {
  description = "name of route table"
  default     = "dom"
}

variable "cidr_block" {
  description = "default route"
  default     = "0.0.0.0/0"
}
