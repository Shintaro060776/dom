resource "aws_api_gateway_rest_api" "text2speech" {
    name = "text2speech"
    description = "API for text2speech"
}

resource "aws_api_gateway_resource" "text2speech" {
    rest_api_id = aws_api_gateway_rest_api.text2speech.id
    parent_id = aws_api_gateway_rest_api.text2speech.root_resource_id
    path_part = "text2speech"
}

resource "aws_api_gateway_method" "text2speech" {
    rest_api_id = aws_api_gateway_rest_api.text2speech.id
    resource_id = aws_api_gateway_resource.text2speech.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "text2speech" {
    rest_api_id = aws_api_gateway_rest_api.text2speech.id
    resource_id = aws_api_gateway_resource.text2speech.id
    http_method = aws_api_gateway_method.text2speech.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.text2speech.invoke_arn
}

resource "aws_api_gateway_deployment" "text2speech" {
    depends_on = [
        aws_api_gateway_integration.text2speech
    ]

    rest_api_id = aws_api_gateway_rest_api.text2speech.id
    description = "initial deployment"
}

resource "aws_api_gateway_stage" "text2speech" {
    deployment_id = aws_api_gateway_deployment.text2speech.id
    rest_api_id = aws_api_gateway_rest_api.text2speech.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "text2speech" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.text2speech.function_name
    principal = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.text2speech.execution_arn}/prod/*/text2speech"
}

resource "aws_api_gateway_rest_api_policy" "text2speech" {
    rest_api_id = aws_api_gateway_rest_api.text2speech.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.text2speech.execution_arn}/*"
            }
        ],
    })
}