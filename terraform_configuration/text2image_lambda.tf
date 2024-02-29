resource "aws_lambda_layer_version" "requests_layer" {
    filename = "/home/runner/work/dom/dom/mylambda_layer.zip"
    layer_name = "requests_layer"

    compatible_runtimes = ["python3.11"]
}


resource "aws_lambda_function" "text2image_lambda" {
  function_name = "text2image_lambda_lambda_function"
  runtime       = "python3.11"
  handler       = "lambda_function.lambda_handler"

  filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"

  role = aws_iam_role.text2image_lambda_role.arn

  timeout = 900

  layers = [
    aws_lambda_layer_version.requests_layer.arn,
  ]

  environment {
    variables = {
      STABILITY_API_KEY = "/myapp/image2video/stability_api_key"
      S3_BUCKET_NAME    = "text2image20090317",
      SLACK_WEBHOOK_URL = data.aws_ssm_parameter.slack_webhook.value
    }
  }
}

resource "aws_iam_role" "text2image_lambda_role" {
  name = "text2image_lambda_role"

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

resource "aws_iam_role_policy_attachment" "text2image_lambda_basic_execution" {
  role       = aws_iam_role.text2image_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "text2image_lambda_s3_access" {
  role       = aws_iam_role.text2image_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess" 
}
