resource "aws_api_gateway_rest_api" "stabilityai" {
    name = "stabilityai"
    description = "stabilityai"
}

resource "aws_api_gateway_resource" "stabilityai" {
    rest_api_id = aws_api_gateway_rest_api.stabilityai.id
    parent_id = aws_api_gateway_rest_api.stabilityai.root_resource_id
    path_part = "stabilityai1"
}

resource "aws_api_gateway_method" "stabilityai" {
    rest_api_id = aws_api_gateway_rest_api.stabilityai.id
    resource_id = aws_api_gateway_resource.stabilityai.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "stabilityai" {
    rest_api_id = aws_api_gateway_rest_api.stabilityai.id
    resource_id = aws_api_gateway_resource.stabilityai.id
    http_method = aws_api_gateway_method.stabilityai.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.stabilityai1.invoke_arn
}

resource "aws_api_gateway_deployment" "stabilityai" {
    depends_on = [
        aws_api_gateway_integration.stabilityai
    ]

    rest_api_id = aws_api_gateway_rest_api.stabilityai.id
}

resource "aws_api_gateway_stage" "stabilityai" {
    deployment_id = aws_api_gateway_deployment.stabilityai.id
    rest_api_id = aws_api_gateway_rest_api.stabilityai.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "stabilityai" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.stabilityai1.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.stabilityai.execution_arn}/prod/*/stabilityai1"
}