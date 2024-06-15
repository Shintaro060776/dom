resource "aws_iam_role" "capture1" {
    name = "capture1"

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

resource "aws_iam_policy" "capture1" {
    name = "capture1"
    description = "capture1"

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "s3:PutObject",
                    "s3:GetObject",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "dynamodb:*"
                ],
                Effect = "Allow",
                Resource = "*"
            },
        ],
    })
}

resource "aws_iam_policy_attachment" "capture1" {
    name = "capture1"
    roles = [aws_iam_role.capture1.name]
    policy_arn = aws_iam_policy.capture1.arn
}

resource "aws_lambda_function" "capture1" {
    function_name = "capture1"

    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"
    handler = "lambda_function.lambda_handler"
    role = aws_iam_role.capture1.arn
    runtime = "python3.11"

    layers = [
        aws_lambda_layer_version.latest_requests_layer.arn,
    ]

    timeout = 900

    environment {
        variables = {
            SLACK_WEBHOOK_URL = data.aws_ssm_parameter.slack_webhook.value
        }
    }
}