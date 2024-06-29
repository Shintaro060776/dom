resource "aws_lambda_function" "prompt_gen" {
    function_name = "prompt_gen_lambda_function"
    runtime = "python3.11"
    handler = "lambda_function.lambda_handler"
    filename = "/path/to/your/lambda_function.zip"
    role = aws_iam_role.prompt_gen_lambda_role.arn
    timeout = 900

    environment {
        variables = {
            STABILITY_API_KEY = "/myapp/image2video/stability_api_key"
            S3_BUCKET_NAME = "prompt-gen-20090317"
            SLACK_WEBHOOK_URL = data.aws_ssm_parameter.slack_webhook.value
        }
    }
}

resource "aws_iam_role" "prompt_gen_lambda_role" {
    name = "prompt_gen_lambda_role"

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
        ]
    })
}

resource "aws_iam_policy" "prompt_gen_lambda_policy" {
    name = "prompt_gen_lambda_policy"

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = ["ssm:GetParameter"],
                Effect = "Allow",
                Resource = "*"
            }
        ]
    })
}

resource "aws_iam_role_policy_attachment" "prompt_gen_lambda_basic_execution" {
    role = aws_iam_role.prompt_gen_lambda_role.name
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSlambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "prompt_gen_lambda_s3_access" {
    role = aws_iam_role.prompt_gen_lambda_role.name
    policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "prompt_gen_lambda_policy_attachment" {
    role = aws_iam_role.prompt_gen_lambda_role.name
    policy_arn = aws_iam_policy.prompt_gen_lambda_policy.arn
}