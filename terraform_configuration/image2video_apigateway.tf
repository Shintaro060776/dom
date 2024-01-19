resource "aws_lambda_permission" "image2video_lambda_permission" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.image2video_lambda.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_api_gateway_rest_api.image2video_api.execution_arn}/prod/POST/image2video"
}

resource "aws_api_gateway_rest_api" "image2video_api" {
  name        = "image2video_api"
  description = "API for image2video application"
}

resource "aws_api_gateway_resource" "image2video_resource" {
  rest_api_id = aws_api_gateway_rest_api.image2video_api.id
  parent_id   = aws_api_gateway_rest_api.image2video_api.root_resource_id
  path_part   = "image2video"
}

resource "aws_api_gateway_method" "image2video_method" {
  rest_api_id   = aws_api_gateway_rest_api.image2video_api.id
  resource_id   = aws_api_gateway_resource.image2video_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "image2video_integration" {
  rest_api_id             = aws_api_gateway_rest_api.image2video_api.id
  resource_id             = aws_api_gateway_resource.image2video_resource.id
  http_method             = aws_api_gateway_method.image2video_method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.image2video_lambda.invoke_arn
}

resource "aws_api_gateway_deployment" "image2video_deployment" {
  depends_on = [
    aws_api_gateway_integration.image2video_integration
  ]

  rest_api_id = aws_api_gateway_rest_api.image2video_api.id
#   stage_name  = "prod" 
}

resource "aws_api_gateway_stage" "image2video_stage" {
  deployment_id = aws_api_gateway_deployment.image2video_deployment.id
  rest_api_id   = aws_api_gateway_rest_api.image2video_api.id
  stage_name    = "prod" 
}
