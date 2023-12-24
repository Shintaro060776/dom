resource "aws_apigatewayv2_api" "realtime" {
    name = "realtime"
    protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
    api_id = aws_apigatewayv2_api.realtime.id
    integration_type = "AWS_PROXY"
    integration_uri = aws_lambda_function.lambda_function.invoke_arn
}

resource "aws_apigatewayv2_route" "realtime_route" {
    api_id = aws_apigatewayv2_api.realtime.id
    route_key = "ANY /realtime"
    target = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "realtime" {
    api_id = aws_apigatewayv2_api.realtime.id
    name = "realtime"
    auto_deploy = true

    access_log_settings {
        destination_arn = aws_cloudwatch_log_group.api_gw_log_group.arn
        format = "$context.identity.sourceIp - [$context.requestTime] \"$context.httpMethod $context.resourcePath $context.protocol\" $context.status $context.responseLength $context.requestId"
    }
}

resource "aws_cloudwatch_log_group" "api_gw_log_group" {
    name = "/aws/apigateway/realtime"
}

resource "aws_lambda_permission" "api_gateway_permission" {
    statement_id = "AllowExecutionFromAPIGateway"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.lambda_function.function_name
    principal = "apigateway.amazonaws.com"
    source_arn = "${aws_apigatewayv2_api.realtime.execution_arn}/*/*"
}