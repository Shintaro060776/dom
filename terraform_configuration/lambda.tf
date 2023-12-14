resource "aws_iam_role" "lambda_role" {
    name = "lambda_role"

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

resource "aws_iam_role_policy" "lambda_policy" {
    name = "lambda_policy"
    role = aws_iam_role.lambda_role.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "translate:TranslateText",
                    "runtime.sagemaker:InvokeEndpoint",
                    "ec:*",
                    "apigateway:*"
                ],
                Resoure = "*",
                Effect = "Allow",
            },
        ],
    })
}

resource "aws_lambda_function" "my_lambda" {
    function_name = "emotion"
    role = aws_iam_role.lambda_role.arn
    handler = "lambda_function.lambda_handler"

    runtime = "python3.8"

    environment {
        variables = {
            OPENAI_API_KEY = "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"
        }
    }
}

resource "aws_cloudwatch_log_group" "lambda_log_group" {
    name = "/aws/lambda/${aws_lambda_function.my_lambda.function_name}"
}

resource "aws_lambda_layer_version" "requests_layer" {
    filename = "/dom/lambda-layer/requests_layer.zip"
    layer_name = "requests-layer"

    compatible_runtime = ["python3.8"]

    source_code_hash = filebase64sha256("/dm/lambda-layer/requests_layer.zip")
}