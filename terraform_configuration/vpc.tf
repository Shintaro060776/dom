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
  subnet_id = aws_subnet.next2.id
  route_table_id = aws_route_table.next.id
}