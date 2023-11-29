resource "aws_vpc" "next" {
    cidr_block = "172.16.0.0/16"
    enable_dns_support = true
    enable_dbs_hostnames = true

    tags = {
        Name = "next"
    }
}

resource "aws_subnet" "next" {
    vpc_id = aws_vpc.next.id
    cidr_block = "172.16.1.0/24"

    tags = {
        Name = "next"
    }
}

resource "aws_internet_gateway" "next" {
    vpc_id = aws_vpc.next.id

    tags = {
        Name = "next"
    }
}

resource "aws_route_table" "next" {
    vpc_id = aws_vpc.next.id

    route {
        cidr_block = "0.0.0.0/0"
        gateway_id = aws_internet_gateway.next.id
    }

    tags = {
        Name = "next"
    }
}

resource "aws_route_table_association" "next" {
    subnet_id = aws_subnet.next.id
    route_table_id = aws_route_table.next.id
}