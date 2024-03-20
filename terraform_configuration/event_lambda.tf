resource "aws_iam_role" "event" {
    name = "event"

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

resource "aws_iam_policy" "event" {
    name = "event"
    description = "event"

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
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

resource "aws_iam_policy_attachment" "event" {
    name = "event"
    roles = [aws_iam_role.event.name]
    policy_arn = aws_iam_policy.event.arn
}

resource "aws_lambda_function" "event" {
    function_name = "event"
    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"
    handler = "lambda_function.lambda_handler"
    role = aws_iam_role.event.arn
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