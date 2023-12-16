resource "aws_api_gateway_rest_api" "my_api" {
    name = "MyAPI"
    description = "API Gateway for my lambda function"
}

resource "aws_api_gateway_resource" "my_api_resource" {
    rest_api_id = aws_api_gateway_rest_api.my_api.id
    parent_id = aws_api_gateway_rest_api.my_api.root_resource_id
    path_part = "myresource"
}

resource "aws_api_gateway_method" "my_api_method" {
  rest_api_id   = aws_api_gateway_rest_api.my_api.id
  resource_id   = aws_api_gateway_resource.my_api_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.my_api.id
  resource_id = aws_api_gateway_resource.my_api_resource.id
  http_method = aws_api_gateway_method.my_api_method.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.my_lambda.invoke_arn
}

resource "aws_api_gateway_deployment" "my_api_deployment" {
    depends_on = [
        aws_api_gateway_integration.lambda_integration
    ]

    rest_api_id = aws_api_gateway_rest_api.my_api.id
    stage_name = "prod"
}


resource "aws_lambda_permission" "api_gateway_invoke" {
    statement_id = "AllowExecutionFromAPIGateway"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.my_lambda.function_name
    source_arn = "${aws_api_gateway_rest_api.my_api.execution_arn}/*/*/*"
}