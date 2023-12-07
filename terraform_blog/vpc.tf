data "aws_vpc" "next" {
  tags = {
    Name = "next"
  }
}

data "aws_subnet" "next" {
  vpc_id = data.aws_vpc.blog.id
  tags = {
    Name = "next" 
  }
}

data "aws_subnet" "next2" {
  vpc_id = data.aws_vpc.blog.id
  tags = {
    Name = "next2" 
  }
}

data "aws_internet_gateway" "next" {
  filter {
    name   = "attachment.vpc-id"
    values = [data.aws_vpc.next.id]
  }
}

data "aws_route_table" "next" {
  vpc_id = data.aws_vpc.next.id
  tags = {
    Name = "next"
  }
}

data "aws_security_group" "vpc_endpoint" {
  name   = "vpc-endpoint-sg"
  vpc_id = data.aws_vpc.next.id
}

data "aws_security_group" "secrets_manager_vpc_endpoint_sg" {
  name   = "secrets-manager-vpc-endpoint-sg"
  vpc_id = data.aws_vpc.next.id
}

data "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id       = data.aws_vpc.next.id
  service_name = "com.amazonaws.ap-northeast-1.ecr.dkr"
}

data "aws_vpc_endpoint" "ecr_api" {
  vpc_id       = data.aws_vpc.next.id
  service_name = "com.amazonaws.ap-northeast-1.ecr.api"
}

data "aws_vpc_endpoint" "logs" {
  vpc_id       = data.aws_vpc.next.id
  service_name = "com.amazonaws.ap-northeast-1.logs"
}

data "aws_vpc_endpoint" "secrets_manager" {
  vpc_id       = data.aws_vpc.next.id
  service_name = "com.amazonaws.ap-northeast-1.secretsmanager"
}