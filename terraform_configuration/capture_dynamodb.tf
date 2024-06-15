resource "aws_dynamodb_table" "image_metadata" {
    name = "ImageMetadata"
    billing_mode = "PAY_PER_REQUEST"
    hash_key = "id"

    attribute {
        name = "id"
        type = "S"
    }

    attribute {
        name = "timestamp"
        type = "N"
    }

    tags = {
        Name = "ImageMetadata"
        Environment = "production"
    }
}