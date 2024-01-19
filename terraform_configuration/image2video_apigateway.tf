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
  binary_media_types = ["multipart/form-data"]
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
  type                    = "HTTP_PROXY"
  uri                     = aws_lambda_function.image2video_lambda.invoke_arn
  passthrough_behavior    = "WHEN_NO_MATCH"

  request_templates = {
    "multipart/form-data" = jsonencode({
      body    = "$input.body",
      headers = {
        #foreach($header in $input.params().header.keySet())
        "$header" = "$util.escapeJavaScript($input.params().header.get($header))"
        #if($foreach.hasNext),#end
        #end
      },
      method  = "$context.httpMethod",
      params  = {
        #foreach($param in $input.params().path.keySet())
        "$param" = "$util.escapeJavaScript($input.params().path.get($param))"
        #if($foreach.hasNext),#end
        #end
      },
      query   = {
        #foreach($queryParam in $input.params().querystring.keySet())
        "$queryParam" = "$util.escapeJavaScript($input.params().querystring.get($queryParam))"
        #if($foreach.hasNext),#end
        #end
      }
    })
  }
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
