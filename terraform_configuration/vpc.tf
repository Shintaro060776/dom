resource "aws_vpc" "dom" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = var.vpc_name
  }
}

resource "aws_subnet" "dom" {
  vpc_id            = aws_vpc.dom.id
  cidr_block        = var.subnet_cidr[0]
  availability_zone = var.availability_zone1

  map_public_ip_on_launch = true

  tags = {
    Name = var.subnet_names[0]
  }
}

resource "aws_subnet" "dom2" {
  vpc_id            = aws_vpc.dom.id
  cidr_block        = var.subnet_cidr[1]
  availability_zone = var.availability_zone2

  map_public_ip_on_launch = true

  tags = {
    Name = var.subnet_names[1]
  }
}

resource "aws_internet_gateway" "dom" {
  vpc_id = aws_vpc.dom.id

  tags = {
    Name = var.internet_gateway_name
  }
}

resource "aws_route_table" "dom" {
  vpc_id = aws_vpc.dom.id

  route {
    cidr_block = var.cidr_block
    gateway_id = aws_internet_gateway.dom.id
  }

  tags = {
    Name = var.aws_route_table
  }
}

resource "aws_route_table_association" "dom" {
  subnet_id      = aws_subnet.dom.id
  route_table_id = aws_route_table.dom.id
}

resource "aws_security_group" "vpc_sg" {
  name        = "vpc-sg"
  description = "Security group for VPC endpoints"
  vpc_id      = aws_vpc.dom.id

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
