resource "aws_lambda_function" "claim_handler_lambda" {
    function_name = "claim_handler_unique"
    runtime = "python3.8"
    handler = "lambda_function.lambda_handler"
    timeout = 900

    s3_bucket = "claim20090317"
    s3_key = "lambda_function.zip"

    role = aws_iam_role.lambda_exec_unique.arn

    environment {
        variables = {
            SAGEMAKER_ENDPOINT_SENTIMENT = "unique-sagemaker-sentiment-endpoint"
            SAGEMAKER_ENDPOINT_TEXT_GEN  = "unique-sagemaker-text-gen-endpoint"
            SLACK_WEBHOOK_URL            = data.aws_ssm_parameter.slack_webhook.value
            DYNAMODB_TABLE_NAME          = "claim"
        }
    }
}

resource "aws_iam_role" "lambda_exec_unique" {
    name = "lambda_exec_role_unique"

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

resource "aws_iam_role_policy" "lambda_policy_unique" {
    name = "lambda_policy_unique"
    role = aws_iam_role.lambda_exec_unique.id

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
                    "comprehend:DetectDominantLanguage",
                    "dynamodb:PutItem",
                    "dynamodb:GetItem"
                ],
                Effect = "Allow",
                Resource = "*"
            },
        ],
    })
}