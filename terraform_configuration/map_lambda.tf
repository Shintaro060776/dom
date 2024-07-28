resource "aws_lambda_function" "map" {
    function_name = "map_lambda_function"
    runtime = "python3.11"
    handler = "lambda_function.lambda_handler"
    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"
    role = aws_iam_role.map_lambda_role.arn
    timeout = 900

    layers = {
        variables = {
            STABILITY_API_KEY = "/myapp/image2video/stability_api_key"
            S3_BUCKET_NAME = "prompt-gen-20090317"
            SLACK_WEBHOOK_URL = data.aws_ssm_parameter.slack_webhook.value
        }
    }
}

resource "aws_iam_role" "map_lambda_role" {
    name = "map_lambda_role"

    assume_role_policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = "sts:AssumeRole",
                Effect = "Allow",
                Principal = {Service = "lambda.amazonaws.com"}
            },
        ]
    })
}

data "aws_iam_policy" "lambda_basic_execution" {
    arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "map_lambda_basic_execution" {
    role = aws_iam_role.map_lambda_role.name
    policy_arn = data.aws_iam_policy.lambda_basic_execution.arn
    depends_on = [aws_iam_role.map_lambda_role]
}

resource "aws_iam_role_policy_attachment" "map_lambda_s3_access" {
    role = aws_iam_role.map_lambda_role.name
    policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    depends_on = [aws_iam_role.map_lambda_role]
}

resource "aws_iam_policy" "map_lambda_policy" {
    name = "map_lambda_policy"

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = ["ssm:GetParameter"],
                Effect = "Allow",
                Resource = "*"
            }
        ]
    })
}

resource "aws_iam_role_policy_attachment" "map_lambda_policy_attachment" {
    role = aws_iam_role.map_lambda_role.name
    policy_arn = aws_iam_policy.map_lambda_policy.arn
    depends_on = [aws_iam_role.map_lambda_role, aws_iam_policy.map_lambda_policy]
}