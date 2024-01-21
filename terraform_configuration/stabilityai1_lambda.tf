resource "aws_iam_role" "lambda_execution_role_stabilityai1" {
    name = "unique-lambda-execution-role-stabilityai1"

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

resource "aws_iam_policy" "lambda_policy_stabilityai1" {
    name = "unique-lambda-policy-stabilityai1"

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
                    "lambda:InvokeFunction"
                ],
                Effect = "Allow",
                Resource = "*"
            },
        ],
    })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attachment_stabilityai1" {
    role = aws_iam_role.lambda_execution_role_stabilityai1.name
    policy_arn = aws_iam_policy.lambda_policy_stabilityai1.arn
}

resource "aws_lambda_function" "stabilityai1" {
    function_name = "stabilityai1"

    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"

    handler = "lambda_function.lambda_handler"

    role = aws_iam_role.lambda_execution_role_stabilityai1.arn
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
            STABILITY_API_KEY = "test"
        }
    }
}