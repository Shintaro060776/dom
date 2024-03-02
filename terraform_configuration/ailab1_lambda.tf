resource "aws_iam_role" "ailab1" {
    name = "ailab1"

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

resource "aws_iam_policy" "ailab1" {
    name = "ailab1"
    description = "ailab1"

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "s3:PutObject",
                    "s3:GetObject",
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

resource "aws_iam_role_policy_attachment" "ailab1" {
    role = aws_iam_role.ailab1.name
    policy_arn = aws_iam_policy.ailab1.arn
}

data "aws_ssm_parameter" "ailab_api_key" {
  name = "/ailab/apiKey"
}

resource "aws_lambda_function" "ailab1" {
    function_name = "ailab1"

    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"

    handler = "lambda_function.lambda_handler"

    role = aws_iam_role.ailab1.arn
    runtime = "python3.11"

    layers = [
        aws_lambda_layer_version.latest_requests_layer.arn,
    ]

    timeout = 900

    environment {
        variables = {
            AILAB_API_KEY = data.aws_ssm_parameter.ailab_api_key.value,
            SLACK_WEBHOOK_URL = data.aws_ssm_parameter.slack_webhook.value
        }
    }
}

