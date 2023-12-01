resource "aws_vpc" "next" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = var.vpc_name
  }
}

resource "aws_subnet" "next" {
  vpc_id            = aws_vpc.next.id
  cidr_block        = var.subnet_cidr[0]
  availability_zone = var.availability_zone1

  map_public_ip_on_launch = true

  tags = {
    Name = var.subnet_names[0]
  }
}

resource "aws_subnet" "next2" {
  vpc_id            = aws_vpc.next.id
  cidr_block        = var.subnet_cidr[1]
  availability_zone = var.availability_zone2

  map_public_ip_on_launch = true

  tags = {
    Name = var.subnet_names[1]
  }
}

resource "aws_internet_gateway" "next" {
  vpc_id = aws_vpc.next.id

  tags = {
    Name = var.internet_gateway_name
  }
}

resource "aws_route_table" "next" {
  vpc_id = aws_vpc.next.id

  route {
    cidr_block = var.cidr_block
    gateway_id = aws_internet_gateway.next.id
  }

  tags = {
    Name = var.aws_route_table
  }
}

resource "aws_route_table_association" "next" {
  subnet_id      = aws_subnet.next.id
  route_table_id = aws_route_table.next.id
}

resource "aws_route_table_association" "next2" {
  subnet_id      = aws_subnet.next2.id
  route_table_id = aws_route_table.next.id
}

resource "aws_security_group" "vpc_endpoint" {
  name        = "vpc-endpoint-sg"
  description = "Security group for VPC endpoints"
  vpc_id      = aws_vpc.next.id

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

resource "aws_security_group" "secrets_manager_vpc_endpoint_sg" {
  name        = "secrets-manager-vpc-endpoint-sg"
  description = "Security group for Secrets Manager VPC Endpoint"
  vpc_id      = aws_vpc.next.id

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
    Name = "secrets-manager-vpc-endpoint-sg"
  }
}

resource "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id            = aws_vpc.next.id
  service_name      = "com.amazonaws.ap-northeast-1.ecr.dkr"
  vpc_endpoint_type = "Interface"

  subnet_ids          = [aws_subnet.next.id, aws_subnet.next2.id]
  private_dns_enabled = true

  security_group_ids = [aws_security_group.vpc_endpoint.id]
}

resource "aws_vpc_endpoint" "ecr_api" {
  vpc_id            = aws_vpc.next.id
  service_name      = "com.amazonaws.ap-northeast-1.ecr.api"
  vpc_endpoint_type = "Interface"

  subnet_ids          = [aws_subnet.next.id, aws_subnet.next2.id]
  private_dns_enabled = true

  security_group_ids = [aws_security_group.vpc_endpoint.id]
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.next.id
  service_name      = "com.amazonaws.ap-northeast-1.s3"
  vpc_endpoint_type = "Gateway"

  route_table_ids = [aws_route_table.next.id]

  tags = {
    Name = "s3-vpc-endpoint"
  }
}

resource "aws_vpc_endpoint" "logs" {
  vpc_id            = aws_vpc.next.id
  service_name      = "com.amazonaws.ap-northeast-1.logs"
  vpc_endpoint_type = "Interface"

  subnet_ids          = [aws_subnet.next.id, aws_subnet.next2.id]
  private_dns_enabled = true

  security_group_ids = [aws_security_group.vpc_endpoint.id]
}

resource "aws_vpc_endpoint" "secrets_manager" {
  vpc_id             = aws_vpc.next.id
  service_name       = "com.amazonaws.ap-northeast-1.secretsmanager"
  vpc_endpoint_type  = "Interface"
  private_dns_enabled = true
  security_group_ids = [aws_security_group.secrets_manager_vpc_endpoint_sg.id]
  subnet_ids         = [aws_subnet.next.id, aws_subnet.next2.id]

  tags = {
    Name = "secrets-manager-vpc-endpoint"
  }
}

# resource "aws_route" "s3_endpoint_route" {
#   route_table_id         = aws_route_table.next.id
#   destination_cidr_block = "0.0.0.0/0"
#   vpc_endpoint_id        = aws_vpc_endpoint.s3.id
#   depends_on             = [aws_vpc_endpoint.s3]
# }