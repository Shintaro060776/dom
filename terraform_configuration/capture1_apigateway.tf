resource "aws_api_gateway_rest_api" "capture1" {
    name        = "capture1"
    description = "API for presigned URL"
}

resource "aws_api_gateway_resource" "capture1" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id
    parent_id   = aws_api_gateway_rest_api.capture1.root_resource_id
    path_part   = "capture1"
}

resource "aws_api_gateway_method" "capture1" {
    rest_api_id   = aws_api_gateway_rest_api.capture1.id
    resource_id   = aws_api_gateway_resource.capture1.id
    http_method   = "ANY"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "capture1" {
    rest_api_id             = aws_api_gateway_rest_api.capture1.id
    resource_id             = aws_api_gateway_resource.capture1.id
    http_method             = aws_api_gateway_method.capture1.http_method
    integration_http_method = "POST"
    type                    = "AWS_PROXY"
    uri                     = aws_lambda_function.capture1.invoke_arn
}

resource "aws_api_gateway_method_response" "capture1_200" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id
    resource_id = aws_api_gateway_resource.capture1.id
    http_method = aws_api_gateway_method.capture1.http_method
    status_code = "200"

    response_parameters = {
        "method.response.header.Access-Control-Allow-Headers" = true,
        "method.response.header.Access-Control-Allow-Methods" = true,
        "method.response.header.Access-Control-Allow-Origin"  = true
    }
}

resource "aws_api_gateway_integration_response" "capture1_200" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id
    resource_id = aws_api_gateway_resource.capture1.id
    http_method = aws_api_gateway_method.capture1.http_method
    status_code = aws_api_gateway_method_response.capture1_200.status_code

    response_parameters = {
        "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
        "method.response.header.Access-Control-Allow-Methods" = "'GET,POST,PUT,DELETE,HEAD,OPTIONS'",
        "method.response.header.Access-Control-Allow-Origin"  = "'*'"
    }
}

resource "aws_api_gateway_method" "capture1_options" {
    rest_api_id   = aws_api_gateway_rest_api.capture1.id
    resource_id   = aws_api_gateway_resource.capture1.id
    http_method   = "OPTIONS"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "capture1_options_integration" {
    rest_api_id             = aws_api_gateway_rest_api.capture1.id
    resource_id             = aws_api_gateway_resource.capture1.id
    http_method             = aws_api_gateway_method.capture1_options.http_method
    integration_http_method = "OPTIONS"
    type                    = "MOCK"

    request_templates = {
        "application/json" = jsonencode({
            statusCode = 200
        })
    }
}

resource "aws_api_gateway_method_response" "capture1_options_200" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id
    resource_id = aws_api_gateway_resource.capture1.id
    http_method = aws_api_gateway_method.capture1_options.http_method
    status_code = "200"

    response_parameters = {
        "method.response.header.Access-Control-Allow-Headers" = true,
        "method.response.header.Access-Control-Allow-Methods" = true,
        "method.response.header.Access-Control-Allow-Origin"  = true
    }
}

resource "aws_api_gateway_integration_response" "capture1_options_integration_200" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id
    resource_id = aws_api_gateway_resource.capture1.id
    http_method = aws_api_gateway_method.capture1_options.http_method
    status_code = aws_api_gateway_method_response.capture1_options_200.status_code

    response_parameters = {
        "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
        "method.response.header.Access-Control-Allow-Methods" = "'GET,POST,PUT,DELETE,HEAD,OPTIONS'",
        "method.response.header.Access-Control-Allow-Origin"  = "'*'"
    }
}

resource "aws_api_gateway_deployment" "capture1" {
    depends_on = [
        aws_api_gateway_integration.capture1,
        aws_api_gateway_integration_response.capture1_200,
        aws_api_gateway_integration.capture1_options_integration,
        aws_api_gateway_integration_response.capture1_options_integration_200
    ]

    rest_api_id = aws_api_gateway_rest_api.capture1.id
    description = "initial deployment"
}

resource "aws_api_gateway_stage" "capture1" {
    deployment_id = aws_api_gateway_deployment.capture1.id
    rest_api_id   = aws_api_gateway_rest_api.capture1.id
    stage_name    = "prod"
}

resource "aws_lambda_permission" "capture1" {
    statement_id  = "AllowAPIGatewayInvoke"
    action        = "lambda:InvokeFunction"
    function_name = aws_lambda_function.capture1.function_name
    principal     = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.capture1.execution_arn}/prod/*/capture1"
}

resource "aws_api_gateway_rest_api_policy" "capture1" {
    rest_api_id = aws_api_gateway_rest_api.capture1.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect    = "Allow",
                Principal = "*",
                Action    = "execute-api:Invoke",
                Resource  = "${aws_api_gateway_rest_api.capture1.execution_arn}/*"
            }
        ],
    })
}