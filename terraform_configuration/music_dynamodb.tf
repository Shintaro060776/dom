resource "aws_dynamodb_table" "MusicInformation" {
    name = "MusicInformation"
    billing_mode = "PAY_PER_REQUEST"
    hash_key = "id"

    attribute {
        name = "id"
        type = "S"
    }

    tags = {
        Name = "MusicInformation"
        Environment = "development"
    }
}