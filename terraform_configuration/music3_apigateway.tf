resource "aws_api_gateway_rest_api" "music3" {
    name = "music3"
    description = "API for Getting All MusicInformation"
}

resource "aws_api_gateway_resource" "music3" {
    rest_api_id = aws_api_gateway_rest_api.music3.id
    parent_id = aws_api_gateway_rest_api.music3.root_resource_id
    path_part = "music3"
}

resource "aws_api_gateway_method" "music3" {
    rest_api_id = aws_api_gateway_rest_api.music3.id
    resource_id = aws_api_gateway_resource.music3.id
    http_method = "ANY"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "music3" {
    rest_api_id = aws_api_gateway_rest_api.music3.id
    resource_id = aws_api_gateway_resource.music3.id
    http_method = aws_api_gateway_method.music3.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.music3.invoke_arn
}

resource "aws_api_gateway_deployment" "music3" {
    depends_on = [
        aws_api_gateway_integration.music3
    ]

    rest_api_id = aws_api_gateway_rest_api.music3.id
    description = "initial deployment"
}

resource "aws_api_gateway_stage" "music3" {
    deployment_id = aws_api_gateway_deployment.music3.id
    rest_api_id = aws_api_gateway_rest_api.music3.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "music3" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.music3.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.music3.execution_arn}/prod/*/music3"
}

resource "aws_api_gateway_rest_api_policy" "music3" {
    rest_api_id = aws_api_gateway_rest_api.music3.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.music3.execution_arn}/*"
            }
        ],
    })
}