resource "aws_lambda_layer_version" "openai_layer" {
    filename = "/home/runner/work/dom/dom/mylayer.zip"
    layer_name = "openai_layer"

    compatible_runtimes = ["python3.11"]
}

resource "aws_iam_role" "lambda_execution_role_smokefree1" {
    name = "smokefree1-lambda-execution-role"

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

resource "aws_iam_policy" "lambda_policy_smokefree1" {
    name = "smokefree1-lambda-policy"

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "s3:Putobject",
                    "s3:GetObject",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutlogEvents",
                    "lambda:InvokeFunction",
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:Query",
                    "dynamodb:Scan"
                ],
                Effect = "Allow",
                Resource = "*"
            },
        ],
    })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attachment_smokefree1" {
    role = aws_iam_role.lambda_execution_role_smokefree1.name
    policy_arn = aws_iam_policy.lambda_policy_smokefree1.arn
}

resource "aws_lambda_function" "smokefree1" {
    function_name = "smokefree1"
    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"

    layers = [
        aws_lambda_layer_version.openai_layer.arn
    ]

    handler = "lambda_function.lambda_handler"
    role = aws_iam_role.lambda_execution_role_smokefree1.arn

    runtime = "python3.11"

    timeout = 900

    environment {
        variables = {
            STABILITY_API_KEY = "test"
            OPENAI_API_KEY = data.aws_ssm_parameter.openai_api_key.value
        }
    }
}