resource "aws_api_gateway_rest_api" "capture2" {
    name = "capture2"
    description = "API for getting drew picture"
}

resource "aws_api_gateway_resource" "capture2" {
    rest_api_id = aws_api_gateway_rest_api.capture2.id
    parent_id = aws_api_gateway_rest_api.capture2.root_resource_id
    path_part = "capture2"
}

resource "aws_api_gateway_method" "capture2" {
    rest_api_id = aws_api_gateway_rest_api.capture2.id
    resource_id = aws_api_gateway_resource.capture2.id
    http_method = "ANY"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "capture2" {
    rest_api_id = aws_api_gateway_rest_api.capture2.id
    resource_id = aws_api_gateway_resource.capture2.id
    http_method = aws_api_gateway_method.capture2.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.capture2.invoke_arn
}

resource "aws_api_gateway_deployment" "capture2" {
    depends_on = [
        aws_api_gateway_integration.capture2
    ]

    rest_api_id = aws_api_gateway_rest_api.capture2.id
    description = "initial deployment"
}

resource "aws_api_gateway_stage" "capture2" {
    deployment_id = aws_api_gateway_deployment.capture2.id
    rest_api_id = aws_api_gateway_rest_api.capture2.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "capture2" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.capture2.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.capture2.execution_arn}/prod/*/capture2"
}

resource "aws_api_gateway_rest_api_policy" "capture2" {
    rest_api_id = aws_api_gateway_rest_api.capture2.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.capture2.execution_arn}/*"
            }
        ],
    })
}