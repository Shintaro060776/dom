resource "aws_api_gateway_rest_api" "event" {
    name = "event"
    description = "API for event application"
}

resource "aws_api_gateway_resource" "event" {
    rest_api_id = aws_api_gateway_rest_api.event.id
    parent_id = aws_api_gateway_rest_api.event.root_resource_id
    path_part = "events"
}

resource "aws_api_gateway_method" "event" {
    rest_api_id = aws_api_gateway_rest_api.event.id
    resource_id = aws_api_gateway_resource.event.id
    http_method = "ANY"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "event" {
    rest_api_id = aws_api_gateway_rest_api.event.id
    resource_id = aws_api_gateway_resource.event.id
    http_method = aws_api_gateway_method.event.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.event.invoke_arn
}

resource "aws_api_gateway_deployment" "event" {
    depends_on = [
        aws_api_gateway_integration.event
    ]

    rest_api_id = aws_api_gateway_rest_api.event.id
    description = "initial deployment"
}

resource "aws_api_gateway_stage" "event" {
    deployment_id = aws_api_gateway_deployment.event.id
    rest_api_id = aws_api_gateway_rest_api.event.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "event" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.event.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.event.execution_arn}/prod/*/events"
}

resource "aws_api_gateway_rest_api_policy" "event" {
    rest_api_id = aws_api_gateway_rest_api.event.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.event.execution_arn}/*"
            }
        ],
    })
}