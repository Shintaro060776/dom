resource "aws_dynamodb_table" "speech" {
    name = "speech"
    billing_mode = "PROVISIONED"
    read_capacity = 5
    write_capacity = 5
    hash_key = "file_key"

    attribute {
        name = "file_key"
        type = "S"
    }

    tags = {
        Name = "speech"
    }
}