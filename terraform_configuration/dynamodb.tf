resource "aws_dynamodb_table" "user_feedback" {
    name = "UserFeedback"
    billing_mode = "PAY_PER_REQUEST"
    hash_key = "UserID"
    range_key = "Timestamp"

    attribute {
        name = "UserID"
        type = "S"
    }

    attribute {
        name = "Timestamp"
        type = "S"
    }

    tags = {
        Name = "UserFeedback"
        Environment = "production"
    }
}