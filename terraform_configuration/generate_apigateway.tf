resource "aws_api_gateway_rest_api" "generate_api" {
    name = "GenerateAPI"
    description = "Rest API for text classification and generation"
}

resource "aws_api_gateway_resource" "generate_resource" {
    rest_api_id = aws_api_gateway_rest_api.generate_api.id
    parent_id = aws_api_gateway_rest_api.generate_api.root_resource_id
    path_part = "generate"
}

resource "aws_api_gateway_method" "generate_method" {
    rest_api_id = aws_api_gateway_rest_api.generate_api.id
    resource_id = aws_api_gateway_resource.generate_resource.id
    http_method = "GET"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "generate_lambda_integration" {
    rest_api_id = aws_api_gateway_rest_api.generate_api.id
    resource_id = aws_api_gateway_resource.generate_resource.id
    http_method = aws_api_gateway_method.generate_method.http_method
    type = "AWS_PROXY"

    uri = aws_lambda_function.generate_lambda.invoke_arn
    integration_http_method = "POST"
}

resource "aws_api_gateway_deployment" "generate_deployment" {
    depends_on = [
        aws_api_gateway_integration.generate_lambda_integration
    ]

    rest_api_id = aws_api_gateway_rest_api.generate_api.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "generate_api_gateway_invoke" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.generate_lambda.function_name
    principal = "apigateway.amazonaws.com"
    source_arn = "${aws_api_gateway_rest_api.generate_api.execution_arn}/*/*/*"
}

resource "aws_cloudwatch_log_group" "generate_log_group" {
    name = "/aws/apigateway/GenerateAPI"

    retention_in_days = 90
}