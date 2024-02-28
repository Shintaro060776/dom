resource "aws_dynamodb_table" "image2image" {
    name = "image2image20090317"
    billing_mode = "PAY_PER_REQUEST"
    hash_key = "FileName"

    attribute {
        name = "FileName"
        type = "S"
    }

    tags = {
        Name = "Image2Image"
    }
}