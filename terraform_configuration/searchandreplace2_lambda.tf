resource "aws_iam_role" "lambda_execution_role_searchandreplace2" {
    name = "searchandreplace2-lambda-execution-role"

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

resource "aws_iam_policy" "lambda_policy_searchandreplace2" {
    name = "searchandreplace2-lambda-policy"

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

resource "aws_iam_role_policy_attachment" "lambda_policy_attachment_searchandreplace2" {
    role = aws_iam_role.lambda_execution_role_searchandreplace2.name
    policy_arn = aws_iam_policy.lambda_policy_searchandreplace2.arn
}

resource "aws_lambda_function" "searchandreplace2" {
    function_name = "searchandreplace2"
    filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"
    handler = "lambda_function.lambda_handler"
    role = aws_iam_role.lambda_execution_role_searchandreplace2.arn

    runtime = "python3.11"

    timeout = 900

    environment {
        variables = {
            STABILITY_API_KEY = "test"
        }
    }
}
