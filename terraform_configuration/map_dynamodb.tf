resource "aws_dynamodb_table" "routes" {
    name = "Routes"
    billing_mode = "PAY_PER_REQUEST"

    hash_key = "userId"
    range_key = "routeId"

    attribute {
        name = "userId"
        type = "S"
    }

    attribute {
        name = "routeId"
        type = "S"
    }

    tags = {
        Name = "RoutesTable"
        Environment = "dev"
    }
}