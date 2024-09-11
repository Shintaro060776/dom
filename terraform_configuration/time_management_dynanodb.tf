resource "aws_dynamodb_table" "user_tasks" {
    name = "UserTasks"
    billing_mode = "PAY_PER_REQUEST"
    hash_key = "user_id"
    range_key = "task_name"

    attribute {
        name = "user_id"
        type = "S"
    }

    attribute {
        name = "task_name"
        type = "S"
    }

    tags = {
        Name = "UserTasks"
        Environment = "prod"
    }
}