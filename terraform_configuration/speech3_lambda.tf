resource "aws_iam_role" "lambda_execution_role_speech3" {
    name = "unique-lambda-execution-role-speech3"

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

resource "aws_iam_policy" "lambda_policy_speech3" {
    name = "unique-lambda-policy-speech3"
    description = "A unique policy for speech3"

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
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

resource "aws_iam_role_policy_attachment" "lambda_policy_attachment_speech3" {
    role = aws_iam_role.lambda_execution_role_speech3.name
    policy_arn = aws_iam_policy.lambda_policy_speech3.arn
}

resource "aws_lambda_function" "speech3" {
    function_name = "speech3"

    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"

    handler = "lambda_function.lambda_handler"

    role = aws_iam_role.lambda_execution_role_speech3.arn
    runtime = "python3.11"

    layers = [
        aws_lambda_layer_version.openai_layer.arn,
        aws_lambda_layer_version.requests_layer.arn,
        aws_lambda_layer_version.additional_request_layer.arn,
        aws_lambda_layer_version.openai_latest_layer.arn,
    ]

    timeout = 900

    emvironment {
        variables = {
            OPENAI_API_KEY = data.aws_ssm_parameter.openai_api_key.value,
            SLACK_WEBHOOK_URL = data.aws_ssm_parameter.slack_webhook.value
        }
    }
}