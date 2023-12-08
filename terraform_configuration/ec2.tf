resource "aws_instance" "dom_ec2" {
    ami = var.ami_id
    instance_type = var.instance_type
    subnet_id = aws_subnet.dom.id
    vpc_security_group_ids = [aws_security_group.vpc_sg.id]
    key_name = aws_key_pair.dom_key.key_name

    tags = {
        Name = "DomEC2Instance"
    }
}

resource "aws_key_pair" "dom_key" {
    key_name = "dom_key"
    public_key = file(var.public_key_path)
}

