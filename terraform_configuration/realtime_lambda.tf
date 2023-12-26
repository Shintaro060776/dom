resource "aws_iam_role" "realtime" {
    name = "lambda_execution_role"

    assume_role_policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = "sts:AssumeRole",
                Effect = "Allow",
                Principal = {
                    Service = "lambda.amazonaws.com"
                },
            },
        ],
    })
}

resource "aws_iam_policy" "realtime" {
    name = "realtime"
    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "sagemaker:*",
                    "cloudwatch:*",
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:UpdateItem",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                Effect = "Allow",
                Resource = "*"
            },
        ],
    })
}

resource "aws_iam_policy_attachment" "realtime" {
    name = "realtime-policy-attachment"
    roles = [aws_iam_role.realtime.name]
    policy_arn = aws_iam_policy.realtime.arn
}

data "aws_ssm_parameter" "slack_webhook" {
    name = "/myapp/slack/webhook"
}

resource "aws_cloudwatch_log_group" "lambda_log_group1" {
    name = "/aws/lambda/${aws_lambda_function.lambda_function.function_name}"
}

resource "aws_lambda_function" "lambda_function" {
    function_name = "realtime"
    s3_bucket = "realtime20090317"
    s3_key = "lambda_function.zip" 

    handler = "lambda_function.lambda_handler"
    runtime = "python3.8"
    timeout = 900
    role = aws_iam_role.realtime.arn

    layers = [aws_lambda_layer_version.requests_layer.arn]

    environment {
        variables = {
            OPENAI_API_KEY = data.aws_ssm_parameter.openai_api_key.value
            SAGEMAKER_ENDPOINT_NAME = "sagemaker-scikit-learn-2023-12-19-11-38-07-739"
            SLACK_WEBHOOK_URL = data.aws_ssm_parameter.slack_webhook.value
        }
    }
}
