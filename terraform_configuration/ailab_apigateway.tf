resource "aws_api_gateway_rest_api" "ailab" {
    name = "ailab"
    description = "API for ailab"
}

resource "aws_api_gateway_resource" "ailab" {
    rest_api_id = aws_api_gateway_rest_api.ailab.id
    parent_id = aws_api_gateway_rest_api.ailab.root_resource_id
    path_part = "ailab"
}

resource "aws_api_gateway_method" "ailab" {
    rest_api_id = aws_api_gateway_rest_api.ailab.id
    resource_id = aws_api_gateway_resource.ailab.id
    http_method = "ANY"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "ailab" {
    rest_api_id = aws_api_gateway_rest_api.ailab.id
    resource_id = aws_api_gateway_resource.ailab.id
    http_method = aws_api_gateway_method.ailab.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.ailab1.invoke_arn
}

resource "aws_api_gateway_deployment" "ailab" {
    depends_on = [
        aws_api_gateway_integration.ailab
    ]

    rest_api_id = aws_api_gateway_rest_api.ailab.id
    description = "initial deployment"
}

resource "aws_api_gateway_stage" "ailab" {
    deployment_id = aws_api_gateway_deployment.ailab.id
    rest_api_id = aws_api_gateway_rest_api.ailab.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "ailab" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.ailab1.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.ailab.execution_arn}/prod/*/ailab"
}

resource "aws_api_gateway_rest_api_policy" "ailab" {
    rest_api_id = aws_api_gateway_rest_api.ailab.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.ailab.execution_arn}/*"
            }
        ],
    })
}