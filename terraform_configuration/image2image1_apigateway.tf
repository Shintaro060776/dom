resource "aws_api_gateway_rest_api" "image2image1" {
  name        = "image2image1"
  description = "API for image2image1"
}

resource "aws_api_gateway_resource" "image2image1" {
  rest_api_id = aws_api_gateway_rest_api.image2image1.id
  parent_id   = aws_api_gateway_rest_api.image2image1.root_resource_id
  path_part   = "image2image1"
}

resource "aws_api_gateway_method" "image2image1" {
  rest_api_id   = aws_api_gateway_rest_api.image2image1.id
  resource_id   = aws_api_gateway_resource.image2image1.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "image2image1" {
  rest_api_id = aws_api_gateway_rest_api.image2image1.id
  resource_id = aws_api_gateway_resource.image2image1.id
  http_method = aws_api_gateway_method.image2image1.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.image2image1.invoke_arn
}

resource "aws_api_gateway_deployment" "image2image1" {
  depends_on = [
    aws_api_gateway_integration.image2image1
  ]

  rest_api_id = aws_api_gateway_rest_api.image2image1.id
}

resource "aws_api_gateway_stage" "image2image1" {
  deployment_id = aws_api_gateway_deployment.image2image1.id
  rest_api_id   = aws_api_gateway_rest_api.image2image1.id
  stage_name    = "prod"
}

resource "aws_lambda_permission" "image2image1" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.image2image1.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_api_gateway_rest_api.image2image1.execution_arn}/prod/*/image2image1"
}

resource "aws_api_gateway_rest_api_policy" "image2image1" {
  rest_api_id = aws_api_gateway_rest_api.image2image1.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Allow",
        Principal = "*",
        Action    = "execute-api:Invoke",
        Resource  = "${aws_api_gateway_rest_api.image2image1.execution_arn}/*"
      }
    ],
  })
}