resource "aws_api_gateway_rest_api" "text2image" {
    name = "text2image"
    description = "API for text2image"
}

resource "aws_api_gateway_resource" "text2image" {
    rest_api_id = aws_api_gateway_rest_api.text2image.id
    parent_id = aws_api_gateway_rest_api.text2image.root_resource_id
    path_part = "text2image"
}

resource "aws_api_gateway_method" "text2image" {
    rest_api_id = aws_api_gateway_rest_api.text2image.id
    resource_id = aws_api_gateway_resource.text2image.id
    http_method = "ANY"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "text2image" {
    rest_api_id = aws_api_gateway_rest_api.text2image.id
    resource_id = aws_api_gateway_resource.text2image.id
    http_method = aws_api_gateway_method.text2image.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.text2image.invoke_arn
}

resource "aws_api_gateway_deployment" "text2image" {
    depends_on = [
        aws_api_gateway_integration.text2image
    ]

    rest_api_id = aws_api_gateway_rest_api.text2image.id
    description = "initial deployment"
}

resource "aws_api_gateway_stage" "text2image" {
    deployment_id = aws_api_gateway_deployment.text2image.id
    rest_api_id = aws_api_gateway_rest_api.text2image.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "text2image" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.text2image.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.text2image.execution_arn}/prod/*/text2image"
}

resource "aws_api_gateway_rest_api_policy" "text2image" {
    rest_api_id = aws_api_gateway_rest_api.text2image.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.text2image.execution_arn}/*"
            }
        ],
    })
}