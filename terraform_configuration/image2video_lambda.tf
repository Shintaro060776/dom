resource "aws_lambda_function" "image2video_lambda" {
  function_name = "image2video_lambda_function"
  runtime       = "python3.11"
  handler       = "lambda_function.lambda_handler"

  filename = "/home/runner/work/dom/dom/dalle/lambda_function.zip"

  role = aws_iam_role.image2video_lambda_role.arn

  timeout = 900

  # layers = [
  #   aws_lambda_layer_version.openai_layer.arn,
  #   aws_lambda_layer_version.requests_layer.arn,
  #   aws_lambda_layer_version.additional_request_layer.arn,
  #   aws_lambda_layer_version.openai_latest_layer.arn,
  # ]

  environment {
    variables = {
      STABILITY_API_KEY = "/myapp/image2video/stability_api_key"
      S3_BUCKET_NAME    = "image2video20090317",
      SLACK_WEBHOOK_URL = data.aws_ssm_parameter.slack_webhook.value
    }
  }
}

resource "aws_iam_role" "image2video_lambda_role" {
  name = "image2video_lambda_role"

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

resource "aws_iam_policy" "image2video_ssm_policy" {
    name = "image2video_ssm_policy"

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

resource "aws_iam_role_policy_attachment" "image2video_lambda_ssm_attachment" {
    role = aws_iam_role.image2video_lambda_role.name
    policy_arn = aws_iam_policy.image2video_ssm_policy.arn
}

resource "aws_iam_role_policy_attachment" "image2video_lambda_basic_execution" {
  role       = aws_iam_role.image2video_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "image2video_lambda_s3_access" {
  role       = aws_iam_role.image2video_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess" 
}
