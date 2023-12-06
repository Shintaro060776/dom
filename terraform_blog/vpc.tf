resource "aws_vpc" "blog" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = var.vpc_name
  }
}

resource "aws_subnet" "blog" {
  vpc_id            = aws_vpc.blog.id
  cidr_block        = var.subnet_cidr[0]
  availability_zone = var.availability_zone1

  map_public_ip_on_launch = true

  tags = {
    Name = var.subnet_names[0]
  }
}

resource "aws_subnet" "blog2" {
  vpc_id            = aws_vpc.blog.id
  cidr_block        = var.subnet_cidr[1]
  availability_zone = var.availability_zone2

  map_public_ip_on_launch = true

  tags = {
    Name = var.subnet_names[1]
  }
}

resource "aws_internet_gateway" "blog" {
  vpc_id = aws_vpc.blog.id

  tags = {
    Name = var.internet_gateway_name
  }
}

resource "aws_route_table" "blog" {
  vpc_id = aws_vpc.blog.id

  route {
    cidr_block = var.cidr_block
    gateway_id = aws_internet_gateway.blog.id
  }

  tags = {
    Name = var.aws_route_table
  }
}

resource "aws_route_table_association" "blog" {
  subnet_id      = aws_subnet.blog.id
  route_table_id = aws_route_table.blog.id
}

resource "aws_route_table_association" "blog2" {
  subnet_id      = aws_subnet.blog2.id
  route_table_id = aws_route_table.blog.id
}

resource "aws_security_group" "vpc_endpoint_blog" {
  name        = "vpc-endpoint-sg-blog"
  description = "Security group for VPC endpoints"
  vpc_id      = aws_vpc.blog.id

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "secrets_manager_vpc_endpoint_sg_blog" {
  name        = "secrets-manager-vpc-endpoint-sg-blog"
  description = "Security group for Secrets Manager VPC Endpoint"
  vpc_id      = aws_vpc.blog.id

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "secrets-manager-vpc-endpoint-sg-blog"
  }
}

resource "aws_vpc_endpoint" "ecr_dkr_blog" {
  vpc_id            = aws_vpc.blog.id
  service_name      = "com.amazonaws.ap-northeast-1.ecr.dkr"
  vpc_endpoint_type = "Interface"

  subnet_ids          = [aws_subnet.blog.id, aws_subnet.blog2.id]
  private_dns_enabled = true

  security_group_ids = [aws_security_group.vpc_endpoint_blog.id]
}

resource "aws_vpc_endpoint" "ecr_api_blog" {
  vpc_id            = aws_vpc.blog.id
  service_name      = "com.amazonaws.ap-northeast-1.ecr.api"
  vpc_endpoint_type = "Interface"

  subnet_ids          = [aws_subnet.blog.id, aws_subnet.blog2.id]
  private_dns_enabled = true

  security_group_ids = [aws_security_group.vpc_endpoint_blog.id]
}

resource "aws_vpc_endpoint" "s3_blog" {
  vpc_id            = aws_vpc.blog.id
  service_name      = "com.amazonaws.ap-northeast-1.s3"
  vpc_endpoint_type = "Gateway"

  route_table_ids = [aws_route_table.blog.id]

  tags = {
    Name = "s3-vpc-endpoint_blog"
  }
}

resource "aws_vpc_endpoint" "logs_blog" {
  vpc_id            = aws_vpc.blog.id
  service_name      = "com.amazonaws.ap-northeast-1.logs"
  vpc_endpoint_type = "Interface"

  subnet_ids          = [aws_subnet.blog.id, aws_subnet.blog2.id]
  private_dns_enabled = true

  security_group_ids = [aws_security_group.vpc_endpoint_blog.id]
}

resource "aws_vpc_endpoint" "secrets_manager" {
  vpc_id             = aws_vpc.blog.id
  service_name       = "com.amazonaws.ap-northeast-1.secretsmanager"
  vpc_endpoint_type  = "Interface"
  private_dns_enabled = true
  security_group_ids = [aws_security_group.secrets_manager_vpc_endpoint_sg_blog.id]
  subnet_ids         = [aws_subnet.blog.id, aws_subnet.blog2.id]

  tags = {
    Name = "secrets-manager-vpc-endpoint_blog"
  }
}

# resource "aws_route" "s3_endpoint_route" {
#   route_table_id         = aws_route_table.next.id
#   destination_cidr_block = "0.0.0.0/0"
#   vpc_endpoint_id        = aws_vpc_endpoint.s3.id
#   depends_on             = [aws_vpc_endpoint.s3]
# }