resource "aws_iam_role" "gpt4_lambda_role" {
  name = "gpt4_lambda_role_unique_name" 

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      },
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution_gpt4" {
    role = aws_iam_role.gpt4_lambda_role.name
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}


resource "aws_lambda_function" "gpt4_lambda_function" {
  function_name = "gpt4_unique_lambda_function_name"

  filename      = "/home/runner/work/dom/dom/gpt4/lambda_function.zip"
  handler       = "lambda_function.lambda_handler"
  role          = aws_iam_role.gpt4_lambda_role.arn
  runtime       = "python3.11"

  timeout       = 900

  layers = [
    aws_lambda_layer_version.openai_layer.arn,
    aws_lambda_layer_version.requests_layer.arn,
    aws_lambda_layer_version.additional_request_layer.arn
    ]

  environment {
    variables = {
      OPENAI_API_KEY = data.aws_ssm_parameter.openai_api_key.value
      SLACK_WEBHOOK_URL            = data.aws_ssm_parameter.slack_webhook.value
    }
  }
}

resource "aws_cloudwatch_log_group" "gpt4_lambda_log_group" {
  name              = "/aws/lambda/${aws_lambda_function.gpt4_lambda_function.function_name}"
  retention_in_days = 14 
}

