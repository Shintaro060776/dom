resource "aws_iam_role" "lambda_role_generate" {
    name = "lambda_role_generate"

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

resource "aws_iam_role_policy" "lambda_policy_generate" {
    name = "lambda_policy_generate"
    role = aws_iam_role.lambda_role_generate.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "sagemaker:InvokeEndpoint",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "s3:GetObject",
                    "translate:TranslateText",
                    "comprehend:DetectDominantLanguage"
                ],
                Effect = "Allow",
                Resource = "*"
            },
        ],
    })
}

resource "aws_lambda_function" "generate_lambda" {
    function_name = "generate_text_classification"
    role = aws_iam_role.lambda_role_generate.arn
    timeout = 900
    handler = "lambda_function.lambda_handler"
    runtime = "python3.8"

    s3_bucket = "well-generate20090317"
    s3_key = "lambda_function.zip"

    environment {
        variables = {
            CATEGORY_CLASSIFICATION_ENDPOINT = "category_endpoint",
            TEXT_GENERATION_ENDPOINT = "text-generation"
        }
    }
}