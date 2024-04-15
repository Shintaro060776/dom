resource "aws_dynamodb_table" "searchandreplace_table" {
    name = "searchandreplace20090317"
    billing_mode = "PAY_PER_REQUEST"
    hash_key = "FileName"

    attribute {
        name = "FileName"
        type = "S"
    }

    tags = {
        Environment = "production"
        Name = "searchandreplace20090317"
    }
}