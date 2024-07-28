resource "aws_api_gateway_rest_api" "map" {
    name = "map"
    description = "API for map"
}

resource "aws_api_gateway_resource" "map" {
    rest_api_id = aws_api_gateway_rest_api.map.id
    parent_id = aws_api_gateway_rest_api.map.root_resource_id
    path_part = "map"
}

resource "aws_api_gateway_method" "map" {
    rest_api_id = aws_api_gateway_rest_api.map.id
    resource_id = aws_api_gateway_resource.map.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "map" {
    rest_api_id = aws_api_gateway_rest_api.map.id
    resource_id = aws_api_gateway_resource.map.id
    http_method = aws_api_gateway_method.map.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.map.invoke_arn
}

resource "aws_api_gateway_deployment" "map" {
    depends_on = [
        aws_api_gateway_integration.map
    ]

    rest_api_id = aws_api_gateway_rest_api.map.id
    description = "Initial Deployment"
}

resource "aws_api_gateway_stage" "map" {
    deployment_id = aws_api_gateway_deployment.map.id
    rest_api_id = aws_api_gateway_rest_api.map.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "map" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.map.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.map.execution_arn}/prod/*/map"
}

resource "aws_api_gateway_rest_api_policy" "map" {
    rest_api_id = aws_api_gateway_rest_api.map.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.map.execution_arn}/*"
            }
        ],
    })
}