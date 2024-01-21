resource "aws_api_gateway_rest_api" "video_status_api" {
  name        = "VideoStatusAPI"
  description = "API for checking video generation status"
}

resource "aws_api_gateway_resource" "video_status_resource" {
  rest_api_id = aws_api_gateway_rest_api.video_status_api.id
  parent_id   = aws_api_gateway_rest_api.video_status_api.root_resource_id
  path_part   = "check_video_status"
}

resource "aws_api_gateway_method" "video_status_method" {
  rest_api_id   = aws_api_gateway_rest_api.video_status_api.id
  resource_id   = aws_api_gateway_resource.video_status_resource.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "video_status_integration" {
  rest_api_id = aws_api_gateway_rest_api.video_status_api.id
  resource_id = aws_api_gateway_resource.video_status_resource.id
  http_method = aws_api_gateway_method.video_status_method.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.stabilityai3.invoke_arn
}

resource "aws_api_gateway_deployment" "video_status_deployment" {
  depends_on = [
    aws_api_gateway_integration.video_status_integration
  ]

  rest_api_id = aws_api_gateway_rest_api.video_status_api.id
  stage_name  = "prod"
}

resource "aws_lambda_permission" "video_status_lambda_permission" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.stabilityai3.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_api_gateway_rest_api.video_status_api.execution_arn}/prod/check_video_status/*"
}