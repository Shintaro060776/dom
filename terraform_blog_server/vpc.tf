data "aws_vpc" "blog" {
  tags = {
    Name = "blog"
  }
}

data "aws_subnet" "blog" {
  vpc_id = data.aws_vpc.blog.id
  tags = {
    Name = "blog" 
  }
}

data "aws_subnet" "blog2" {
  vpc_id = data.aws_vpc.blog.id
  tags = {
    Name = "blog2" 
  }
}

data "aws_internet_gateway" "blog" {
  filter {
    name   = "attachment.vpc-id"
    values = [data.aws_vpc.blog.id]
  }
}

data "aws_route_table" "blog" {
  vpc_id = data.aws_vpc.blog.id
  tags = {
    Name = "blog"
  }
}

data "aws_security_group" "vpc_endpoint_blog" {
  name   = "vpc-endpoint-sg-blog"
  vpc_id = data.aws_vpc.blog.id
}

data "aws_security_group" "secrets_manager_vpc_endpoint_sg_blog" {
  name   = "secrets-manager-vpc-endpoint-sg-blog"
  vpc_id = data.aws_vpc.blog.id
}

data "aws_vpc_endpoint" "ecr_dkr_blog" {
  vpc_id       = data.aws_vpc.blog.id
  service_name = "com.amazonaws.ap-northeast-1.ecr.dkr"
}

data "aws_vpc_endpoint" "ecr_api_blog" {
  vpc_id       = data.aws_vpc.blog.id
  service_name = "com.amazonaws.ap-northeast-1.ecr.api"
}

data "aws_vpc_endpoint" "logs_blog" {
  vpc_id       = data.aws_vpc.blog.id
  service_name = "com.amazonaws.ap-northeast-1.logs"
}

data "aws_vpc_endpoint" "secrets_manager" {
  vpc_id       = data.aws_vpc.blog.id
  service_name = "com.amazonaws.ap-northeast-1.secretsmanager"
}