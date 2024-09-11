resource "aws_iam_role" "time_management_role" {
    name = "time_management"

    assume_role_policy = jsonencode({
        Version = "2012-10-17",
        Statement = [{
            Action = "sts:AssumeRole",
            Effect = "Allow",
            Principal = {
                Service = "lambda.amazonaws.com"
            },
        }]
    })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution_time_management" {
    role = aws_iam_role.time_management_role.name
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_dynamodb_access" {
    role = aws_iam_role.time_management_role.name
    policy_arn = "arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess"
}

resource "aws_lambda_function" "time_management_lambda_function" {
    function_name = "time_management"

    filename = "/home/runner/work/dom/dom/gpt4/lambda_function.zip"
    handler = "lambda_function.lambda_handler"
    layer_name = "openai_latest_layer"
    role = aws_iam_role.time_management_role.arn
    runtime = "python3.11"

    timeout = 900

    layers = [
        aws_lambda_layer_version.openai_latest_layer.arn,
    ]

    environment {
        variables = {
            OPENAI_API_KEY = data.aws_ssm_parameter.openai_api_key.value
        }
    }
}

resource "aws_cloudwatch_log_group" "time_management_lambda_log_group" {
    name = "/aws/lambda/${aws_lambda_function.time_management_lambda_function.function_name}"
    retention_in_days = 14
}