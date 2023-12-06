data "aws_vpc" "selected" {
  tags = {
    Name = "blog"
  }
}

output "vpc_id" {
  value = data.aws_vpc.selected.id
}

resource "aws_subnet" "blog_server" {
  vpc_id            = data.aws_vpc.selected.id
  cidr_block        = var.subnet_cidr[0]
  availability_zone = var.availability_zone1

  map_public_ip_on_launch = true

  tags = {
    Name = var.subnet_names[0]
  }
}

resource "aws_subnet" "blog_server2" {
  vpc_id            = data.aws_vpc.selected.id
  cidr_block        = var.subnet_cidr[1]
  availability_zone = var.availability_zone2

  map_public_ip_on_launch = true

  tags = {
    Name = var.subnet_names[1]
  }
}

resource "aws_internet_gateway" "blog_server" {
  vpc_id = data.aws_vpc.selected.id

  tags = {
    Name = var.internet_gateway_name
  }
}

resource "aws_route_table" "blog_server" {
  vpc_id = data.aws_vpc.selected.id

  route {
    cidr_block = var.cidr_block
    gateway_id = aws_internet_gateway.blog_server.id
  }

  tags = {
    Name = var.aws_route_table
  }
}

resource "aws_route_table_association" "blog_server" {
  subnet_id      = aws_subnet.blog_server.id
  route_table_id = aws_route_table.blog_server.id
}

resource "aws_route_table_association" "blog_server2" {
  subnet_id      = aws_subnet.blog_server2.id
  route_table_id = aws_route_table.blog_server.id
}

resource "aws_security_group" "vpc_endpoint" {
  name        = "vpc-endpoint-sg-blog_server"
  description = "Security group for VPC endpoints"
  vpc_id      = data.aws_vpc.selected.id

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

resource "aws_security_group" "secrets_manager_vpc_endpoint_sg_blog_server" {
  name        = "secrets-manager-vpc-endpoint-sg-blog_server"
  description = "Security group for Secrets Manager VPC Endpoint"
  vpc_id      = data.aws_vpc.selected.id

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
    Name = "secrets-manager-vpc-endpoint-sg-blog_server"
  }
}

resource "aws_vpc_endpoint" "ecr_dkr_blog_server" {
  vpc_id            = data.aws_vpc.selected.id
  service_name      = "com.amazonaws.ap-northeast-1.ecr.dkr"
  vpc_endpoint_type = "Interface"

  subnet_ids          = [aws_subnet.blog_server.id, aws_subnet.blog_server2.id]
  private_dns_enabled = true

  security_group_ids = [aws_security_group.vpc_endpoint.id]
}

resource "aws_vpc_endpoint" "ecr_api_blog_server" {
  vpc_id            = data.aws_vpc.selected.id
  service_name      = "com.amazonaws.ap-northeast-1.ecr.api"
  vpc_endpoint_type = "Interface"

  subnet_ids          = [aws_subnet.blog_server.id, aws_subnet.blog_server.id]
  private_dns_enabled = true

  security_group_ids = [aws_security_group.vpc_endpoint.id]
}

resource "aws_vpc_endpoint" "s3_blog_server" {
  vpc_id            = data.aws_vpc.selected.id
  service_name      = "com.amazonaws.ap-northeast-1.s3"
  vpc_endpoint_type = "Gateway"

  route_table_ids = [aws_route_table.blog_server.id]

  tags = {
    Name = "s3-vpc-endpoint_blog_server"
  }
}

resource "aws_vpc_endpoint" "logs_blog_server" {
  vpc_id            = data.aws_vpc.selected.id
  service_name      = "com.amazonaws.ap-northeast-1.logs"
  vpc_endpoint_type = "Interface"

  subnet_ids          = [aws_subnet.blog_server.id, aws_subnet.blog_server2.id]
  private_dns_enabled = true

  security_group_ids = [aws_security_group.vpc_endpoint.id]
}

resource "aws_vpc_endpoint" "secrets_manager" {
  vpc_id             = data.aws_vpc.selected.id
  service_name       = "com.amazonaws.ap-northeast-1.secretsmanager"
  vpc_endpoint_type  = "Interface"
  private_dns_enabled = true
  security_group_ids = [aws_security_group.secrets_manager_vpc_endpoint_sg.id]
  subnet_ids         = [aws_subnet.blog_server.id, aws_subnet.blog_server.id]

  tags = {
    Name = "secrets-manager-vpc-endpoint_blog_server"
  }
}

# resource "aws_route" "s3_endpoint_route" {
#   route_table_id         = aws_route_table.next.id
#   destination_cidr_block = "0.0.0.0/0"
#   vpc_endpoint_id        = aws_vpc_endpoint.s3.id
#   depends_on             = [aws_vpc_endpoint.s3]
# 