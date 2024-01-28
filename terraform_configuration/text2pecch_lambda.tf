resource "aws_iam_role" "lambda_execution_role_text2speech" {
    name = "text2speech-lambda-execution-role"

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

resource "aws_iam_policy" "lambda_policy_text2speech" {
    name = "text2speech-lambda-policy"

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
                    "lambda:InvokeFunction",
                    "translate:TranslateText"
                ],
                Effect = "Allow",
                Resource = "*"
            },
        ],
    })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attachment_text2speech" {
    role = aws_iam_role.lambda_execution_role_text2speech.name
    policy_arn = aws_iam_policy.lambda_policy_text2speech.arn
}

resource "aws_lambda_function" "text2speech" {
    function_name = "text2speech"

    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"

    handler = "lambda_function.lambda_handler"

    role = aws_iam_role.lambda_execution_role_text2speech.arn
    runtime = "python3.11"

    layers = [
        aws_lambda_layer_version.openai_layer.arn,
        aws_lambda_layer_version.requests_layer.arn,
        aws_lambda_layer_version.additional_request_layer.arn,
        aws_lambda_layer_version.openai_latest_layer.arn,
    ]

    timeout = 900

    environment {
        variables = {
            OPENAI_API_KEY = data.aws_ssm_parameter.openai_api_key.value
        }
    }
}