resource "aws_api_gateway_rest_api" "Speech3" {
    name = "Speech3"
    description = "Speech3"
}

resource "aws_api_gateway_resource" "Speech3" {
    rest_api_id = aws_api_gateway_rest_api.Speech3.id
    parent_id = aws_api_gateway_rest_api.Speech3.root_resource_id
    path_part = "speech3"
}

resource "aws_api_gateway_method" "Speech3" {
    rest_api_id = aws_api_gateway_rest_api.Speech3.id
    resource_id = aws_api_gateway_resource.Speech3.id
    http_method = "GET"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "Speech3" {
    rest_api_id = aws_api_gateway_rest_api.Speech3.id
    resource_id = aws_api_gateway_resource.Speech3.id
    http_method = aws_api_gateway_method.Speech3.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.speech3.invoke_arn
}

resource "aws_api_gateway_deployment" "Speech3" {
    depends_on = [
        aws_api_gateway_integration.Speech3
    ]

    rest_api_id = aws_api_gateway_rest_api.Speech3.id
    # stage_name = "prod"
}

resource "aws_api_gateway_stage" "Speech3" {
    deployment_id = aws_api_gateway_deployment.Speech3.id
    rest_api_id = aws_api_gateway_rest_api.Speech3.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "Speech3" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.speech3.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.Speech3.execution_arn}/prod/GET/speech3"
}