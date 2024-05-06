resource "aws_dynamodb_table" "smokefree_user_data" {
    name = "SmokeFreeUserData"
    billing_mode = "PROVISIONED"
    read_capacity = 10
    write_capacity = 10
    hash_key = "user_id"

    attribute {
        name = "user_id"
        type = "S"
    }

    attribute {
        name = "timestamp"
        type = "S"
    }

    global_secondary_index {
        name = "TimestampIndex"
        hash_key = "timestamp"
        projection_type = "ALL"
        read_capacity = 5
        write_capacity = 5
    }

    tags = {
        Name = "SmokeFreeUserData"
        Environment = "production"
    }
}