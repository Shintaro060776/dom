resource "aws_api_gateway_rest_api" "searchandreplace1" {
    name = "searchandreplace1"
    description = "API for searchandreplace1"
}

resource "aws_api_gateway_resource" "searchandreplace1" {
    rest_api_id = aws_api_gateway_rest_api.searchandreplace1,id
    parent_id = aws_api_gateway_rest_api.searchandreplace1.root_resource_id
    path_part = "searchandreplace1"
}

resource "aws_api_gateway_method" "searchandreplace1" {
    rest_api_id = aws_api_gateway_rest_api.searchandreplace1.id
    resource_id = aws_api_gateway_resource.searchandreplace1.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "searchandreplace1" {
    rest_api_id = aws_api_gateway_rest_api.searchandreplace1.id
    resource_id = aws_api_gateway_resource.searchandreplace1.id
    http_method = aws_api_gateway_method.searchandreplace1.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.searchandreplace1.invoke_arn
}

resource "aws_api_gateway_deployment" "searchandreplace1" {
    depends_on = [
        aws_api_gateway_integration.searchandreplace1
    ]

    rest_api_id = aws_api_gateway_rest_api.searchandreplace1.id
}

resource "aws_api_gateway_stage" "searchandreplace1" {
    deployment_id = aws_api_gateway_deployment.searchandreplace1.id
    rest_api_id = aws_api_gateway_rest_api.searchandreplace1.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "searchandreplace1" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.searchandreplace1.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.searchandreplace1.execution_arn}/prod/*/searchandreplace1"
}

resource "aws_api_gateway_rest_api_policy" "searchandreplace1" {
    rest_api_id = aws_api_gateway_rest_api.searchandreplace1.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.searchandreplace1.execution_arn}/*"
            }
        ],
    })
}
