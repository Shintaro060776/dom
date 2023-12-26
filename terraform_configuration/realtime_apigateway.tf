resource "aws_api_gateway_rest_api" "real_api" {
  name        = "real"
  description = "Real API Gateway API"
}

resource "aws_api_gateway_resource" "real_resource" {
  rest_api_id = aws_api_gateway_rest_api.real_api.id
  parent_id   = aws_api_gateway_rest_api.real_api.root_resource_id
  path_part   = "resource"
}

resource "aws_api_gateway_method" "real_method" {
  rest_api_id   = aws_api_gateway_rest_api.real_api.id
  resource_id   = aws_api_gateway_resource.real_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_real_integration" {
  rest_api_id = aws_api_gateway_rest_api.real_api.id
  resource_id = aws_api_gateway_resource.real_resource.id
  http_method = aws_api_gateway_method.real_method.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.lambda_function.invoke_arn
}

resource "aws_api_gateway_deployment" "real_deployment" {
  depends_on = [aws_api_gateway_integration.lambda_real_integration]

  rest_api_id = aws_api_gateway_rest_api.real_api.id

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_stage" "real_stage" {
  stage_name    = "prod"
  rest_api_id   = aws_api_gateway_rest_api.real_api.id
  deployment_id = aws_api_gateway_deployment.real_deployment.id

  xray_tracing_enabled = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.real_api_logs.arn
    format          = "json"
  }

  cloudwatch_logs_role_arn = aws_iam_role.api_gateway_cloudwatch_role_real.arn
}

resource "aws_iam_role" "api_gateway_cloudwatch_role_real" {
  name = "api_gateway_cloudwatch_role_real"

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

resource "aws_iam_role_policy" "api_gateway_cloudwatch_policy_real" {
  name = "api_gateway_cloudwatch_policy_real"
  role = aws_iam_role.api_gateway_cloudwatch_role_real.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ],
        Effect = "Allow",
        Resource = "*"
      },
    ],
  })
}

resource "aws_cloudwatch_log_group" "real_api_logs" {
  name = "/aws/apigateway/real"
}

resource "aws_lambda_permission" "real_api_gateway_permission" {
  statement_id  = "AllowExecutionFromRealAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_function.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_api_gateway_rest_api.real_api.execution_arn}/*/*/*"
}