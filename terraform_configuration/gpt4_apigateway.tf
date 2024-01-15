resource "aws_api_gateway_rest_api" "gpt4_api_gateway" {
    name = "gpt4-api-gateway-unique-name"
    description = "API Gateway for GPT4 Lambda Function"
}

resource "aws_api_gateway_resource" "gpt4_api_resource" {
    rest_api_id = aws_api_gateway_rest_api.gpt4_api_gateway.id
    parent_id = aws_api_gateway_rest_api.gpt4_api_gateway.root_resource_id
    path_part = "gpt4path"
}

resource "aws_api_gateway_method" "gpt4_api_method" {
    rest_api_id = aws_api_gateway_rest_api.gpt4_api_gateway.id
    resource_id = aws_api_gateway_resource.gpt4_api_resource.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "gpt4_lambda_integration" {
    rest_api_id = aws_api_gateway_rest_api.gpt4_api_gateway.id
    resource_id = aws_api_gateway_resource.gpt4_api_resource.id
    http_method = aws_api_gateway_method.gpt4_api_method.http_method
    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.gpt4_lambda_function.invoke_arn
}

resource "aws_api_gateway_deployment" "gpt4_api_deployment" {
    depends_on = [
        aws_api_gateway_integration.gpt4_lambda_integration
    ]

    rest_api_id = aws_api_gateway_rest_api.gpt4_api_gateway.id
    stage_name = "prod"
}

resource "aws_api_gateway_stage" "gpt4_api_stage" {
    stage_name = aws_api_gateway_deployment.gpt4_api_deployment.stage_name
    rest_api_id = aws_api_gateway_rest_api.gpt4_api_gateway.id
    deployment_id = aws_api_gateway_deployment.gptp4_api_deployment.id

    xray_tracing_enabled = true

    access_log_settings {
        destination_arn = aws_cloudwatch_log_group.gpt4_api_log_group.arn
        format = "{'requestId':'$context.requestId','ip':'$context.identity.sourceIp','requestTime':'$context.requestTime','httpMethod':'$context.httpMethod','status':'$context.status','protocol':'$context.protocol','responseLength':'$context.responseLength'}"
    }
}

resource "aws_cloudwatch_log_group" "gpt4_api_log_group" {
    name =   "/aws/apigateway/${aws_api_gateway_rest_api.gpt4_api_gateway.name}"
    retention_in_days = 7
}

resource "aws_lambda_permission" "gpt4_api_gateway_permission" {
    statement_id = "AllowExecutionFromAPIGateway"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.gpt4_lambda_function.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.gpt4_api_gateway.execution_arn}/*/*"
}
