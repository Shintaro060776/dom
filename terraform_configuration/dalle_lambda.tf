resource "aws_lambda_layer_version" "openai_layer" {
    filename = "/home/runner/work/dom/dom/openai.zip"
    layer_name = "openai_layer"

    compatible_runtimes = ["python3.8"]
}

resource "aws_lambda_function" "openai_image_generator" {
    function_name = "openai_image_generator"
    role = aws_iam_role.lambda_iam_role.arn
    handler = "lambda_function.lambda_handler"
    runtime = "python3.8"
    timeout = 900

    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"

    layers = [
        aws_lambda_layer_version.openai_layer.arn,
        aws_lambda_layer_version.requests_layer.arn
    ]

    environment {
        variables = {
            OPENAI_API_KEY = data.aws_ssm_parameter.openai_api_key.value
            SLACK_WEBHOOK_URL            = data.aws_ssm_parameter.slack_webhook.value
        }
    }
}

resource "aws_iam_role" "lambda_iam_role" {
    name = "lambda_iam_role"

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

resource "aws_cloudwatch_log_group" "dalle_lambda_log_group" {
    name = "/aws/lambda/${aws_lambda_function.openai_image_generator.function_name}"
    retention_in_days = 14
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
    role = aws_iam_role.lambda_iam_role.name
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

