resource "aws_api_gateway_rest_api" "Speech2" {
    name = "Speech2"
    description = "Speech2"
}

resource "aws_api_gateway_resource" "Speech2" {
    rest_api_id = aws_api_gateway_rest_api.Speech2.id
    parent_id = aws_api_gateway_rest_api.Speech2.root_resource_id
    path_part = "speech2"
}

resource "aws_api_gateway_method" "Speech2" {
    rest_api_id = aws_api_gateway_rest_api.Speech2.id
    resource_id = aws_api_gateway_resource.Speech2.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "Speech2" {
    rest_api_id = aws_api_gateway_rest_api.Speech2.id
    resource_id = aws_api_gateway_resource.Speech2.id
    http_method = aws_api_gateway_method.Speech2.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.speech2.invoke_arn
}

resource "aws_api_gateway_deployment" "Speech2" {
    depends_on = [
        aws_api_gateway_integration.Speech2
    ]

    rest_api_id = aws_api_gateway_rest_api.Speech2.id
    # stage_name = "prod"
}

resource "aws_api_gateway_stage" "Speech2" {
    deployment_id = aws_api_gateway_deployment.Speech2.id
    rest_api_id = aws_api_gateway_rest_api.Speech2.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "Speech2" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.speech2.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.Speech2.execution_arn}/*/*/*"
}