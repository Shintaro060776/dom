resource "aws_api_gateway_rest_api" "music1" {
    name = "music1"
    description = "API for presigned URL"
}

resource "aws_api_gateway_resource" "music1" {
    rest_api_id = aws_api_gateway_rest_api.music1.id
    parent_id = aws_api_gateway_rest_api.music1.root_resource_id
    path_part = "music1"
}

resource "aws_api_gateway_method" "music1" {
    rest_api_id = aws_api_gateway_rest_api.music1.id
    resource_id = aws_api_gateway_resource.music1.id
    http_method = "ANY"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "music1" {
    rest_api_id = aws_api_gateway_rest_api.music1.id
    resource_id = aws_api_gateway_resource.music1.id
    http_method = aws_api_gateway_method.music1.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.music1.invoke_arn
}

resource "aws_api_gateway_deployment" "music1" {
    depends_on = [
        aws_api_gateway_integration.music1
    ]

    rest_api_id = aws_api_gateway_rest_api.music1.id
    description = "initial deployment"
}

resource "aws_api_gateway_stage" "music1" {
    deployment_id = aws_api_gateway_deployment.music1.id
    rest_api_id = aws_api_gateway_rest_api.music1.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "music1" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.music1.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.music1.execution_arn}/prod/*/music1"
}

resource "aws_api_gateway_rest_api_policy" "music1" {
    rest_api_id = aws_api_gateway_rest_api.music1.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.music1.execution_arn}/*"
            }
        ],
    })
}