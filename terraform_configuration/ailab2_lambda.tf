resource "aws_iam_role" "ailab2" {
    name = "ailab2"

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

resource "aws_iam_policy" "ailab2" {
    name = "ailab2"
    description = "ailab2"

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

resource "aws_iam_role_policy_attachment" "ailab2" {
    role = aws_iam_role.ailab2.name
    policy_arn = aws_iam_policy.ailab2.arn
}

resource "aws_lambda_function" "ailab2" {
    function_name = "ailab2"

    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"

    handler = "lambda_function.lambda_handler"

    role = aws_iam_role.ailab2.arn
    runtime = "python3.11"

    timeout = 900

    layers = [
        aws_lambda_layer_version.latest_requests_layer.arn,
    ]

    environment {
        variables = {
            AILAB_API_KEY = data.aws_ssm_parameter.ailab_api_key.value,
            SLACK_WEBHOOK_URL = data.aws_ssm_parameter.slack_webhook.value
        }
    }
}

