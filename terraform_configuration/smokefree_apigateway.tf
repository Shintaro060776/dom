resource "aws_api_gateway_rest_api" "smokefree1" {
    name = "smokefree1"
    description = "API for smokefree1"
}

resource "aws_api_gateway_resource" "smokefree1" {
    rest_api_id = aws_api_gateway_rest_api.smokefree1.id
    parent_id = aws_api_gateway_rest_api.smokefree1.root_resource_id
    path_part = "smokefree1"
}

resource "aws_api_gateway_method" "smokefree1" {
    rest_api_id = aws_api_gateway_rest_api.smokefree1.id
    resource_id = aws_api_gateway_resource.smokefree1.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "smokefree1" {
    rest_api_id = aws_api_gateway_rest_api.smokefree1.id
    resource_id = aws_api_gateway_resource.smokefree1.id
    http_method = aws_api_gateway_method.smokefree1.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.smokefree1.invoke_arn
}

resource "aws_api_gateway_deployment" "smokefree1" {
    depends_on = [
        aws_api_gateway_integration.smokefree1
    ]

    rest_api_id = aws_api_gateway_rest_api.smokefree1.id
}

resource "aws_api_gateway_stage" "smokefree1" {
    deployment_id = aws_api_gateway_deployment.smokefree1.id
    rest_api_id = aws_api_gateway_rest_api.smokefree1.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "smokefree1" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.smokefree1.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.smokefree1.execution_arn}/prod/*/smokefree1"
}

resource "aws_api_gateway_rest_api_policy" "smokefree1" {
    rest_api_id = aws_api_gateway_rest_api.smokefree1.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.smokefree1.execution_arn}/*"
            }
        ],
    })
}