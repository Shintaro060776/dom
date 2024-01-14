resource "aws_api_gateway_rest_api" "unique_openai_api" {
  name        = "unique-openai-api"
  description = "API for OpenAI Lambda Function"
}

resource "aws_api_gateway_resource" "unique_openai_resource" {
  rest_api_id = aws_api_gateway_rest_api.unique_openai_api.id
  parent_id   = aws_api_gateway_rest_api.unique_openai_api.root_resource_id
  path_part   = "openai"
}

resource "aws_api_gateway_method" "unique_openai_post" {
  rest_api_id   = aws_api_gateway_rest_api.unique_openai_api.id
  resource_id   = aws_api_gateway_resource.unique_openai_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "unique_openai_lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.unique_openai_api.id
  resource_id = aws_api_gateway_resource.unique_openai_resource.id
  http_method = "POST"
  type        = "AWS_PROXY"
  uri         = aws_lambda_function.openai_image_generator.invoke_arn
}

resource "aws_api_gateway_deployment" "unique_openai_deployment" {
  depends_on = [
    aws_api_gateway_integration.unique_openai_lambda_integration
  ]

  rest_api_id = aws_api_gateway_rest_api.unique_openai_api.id
  stage_name  = "prod"
}

resource "aws_cloudwatch_log_group" "unique_openai_log_group" {
  name = "/aws/apigateway/unique-openai-api"
}

resource "aws_iam_role" "unique_openai_apigw_cloudwatch_role" {
  name = "unique-openai-apigw-cloudwatch-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "apigateway.amazonaws.com"
        },
      },
    ],
  })
}

resource "aws_iam_role_policy" "unique_openai_apigw_cloudwatch_policy" {
  name   = "unique-openai-apigw-cloudwatch-policy"
  role   = aws_iam_role.unique_openai_apigw_cloudwatch_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents"
        ],
        Effect = "Allow",
        Resource = "*"
      },
    ],
  })
}

resource "aws_api_gateway_account" "unique_openai_account" {
  cloudwatch_role_arn = aws_iam_role.unique_openai_apigw_cloudwatch_role.arn
}