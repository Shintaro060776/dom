resource "aws_api_gateway_rest_api" "capture1" {
    name = "capture1"
    description = "API for presigned URL"
}

resource "aws_api_gateway_resource" "capture1" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id
    parent_id = aws_api_gateway_rest_api.capture1.root_resource_id
    path_part = "capture1"
}

resource "aws_api_gateway_method" "capture1" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id
    resource_id = aws_api_gateway_resource.capture1.id
    http_method = "ANY"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "capture1" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id
    resource_id = aws_api_gateway_resource.capture1.id
    http_method = aws_api_gateway_method.capture1.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.capture1.invoke_arn
}

resource "aws_api_gateway_deployment" "capture1" {
    depends_on = [
        aws_api_gateway_integration.capture1
    ]

    rest_api_id = aws_api_gateway_rest_api.capture1.id
    description = "initial deployment"
}

resource "aws_api_gateway_stage" "capture1" {
    deployment_id = aws_api_gateway_deployment.capture1.id
    rest_api_id = aws_api_gateway_rest_api.capture1.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "capture1" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.capture1.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.capture1.execution_arn}/prod/*/capture1"
}

resource "aws_api_gateway_rest_api_policy" "capture1" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.capture1.execution_arn}/*"
            }
        ],
    })
}