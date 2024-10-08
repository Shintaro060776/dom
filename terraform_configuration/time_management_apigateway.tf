resource "aws_api_gateway_rest_api" "time_management_api_gateway" {
    name        = "time_management"
    description = "API Gateway for time management Lambda Function"
}

resource "aws_api_gateway_resource" "time_management_api_resource" {
    rest_api_id = aws_api_gateway_rest_api.time_management_api_gateway.id
    parent_id   = aws_api_gateway_rest_api.time_management_api_gateway.root_resource_id
    path_part   = "timemanagement"
}

# POSTメソッドの設定
resource "aws_api_gateway_method" "time_management_api_method" {
    rest_api_id = aws_api_gateway_rest_api.time_management_api_gateway.id
    resource_id = aws_api_gateway_resource.time_management_api_resource.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "time_management_lambda_integration" {
    rest_api_id             = aws_api_gateway_rest_api.time_management_api_gateway.id
    resource_id             = aws_api_gateway_resource.time_management_api_resource.id
    http_method             = aws_api_gateway_method.time_management_api_method.http_method
    integration_http_method = "POST"
    type                    = "AWS_PROXY"
    uri                     = aws_lambda_function.time_management_lambda_function.invoke_arn
}

# CORS用のOPTIONSメソッドを追加
resource "aws_api_gateway_method" "time_management_options_method" {
    rest_api_id = aws_api_gateway_rest_api.time_management_api_gateway.id
    resource_id = aws_api_gateway_resource.time_management_api_resource.id
    http_method = "OPTIONS"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "time_management_options_integration" {
    rest_api_id             = aws_api_gateway_rest_api.time_management_api_gateway.id
    resource_id             = aws_api_gateway_resource.time_management_api_resource.id
    http_method             = aws_api_gateway_method.time_management_options_method.http_method
    type                    = "MOCK"  # OPTIONSリクエスト用にMOCKレスポンスを使用

    request_templates = {
        "application/json" = "{\"statusCode\": 200}"
    }

    integration_response {
        status_code = "200"

        response_parameters = {
            "method.response.header.Access-Control-Allow-Headers" = "'Content-Type'"
            "method.response.header.Access-Control-Allow-Methods" = "'OPTIONS,POST'"
            "method.response.header.Access-Control-Allow-Origin"  = "'*'"
        }

        response_templates = {
            "application/json" = ""
        }
    }
}

# CORS用のレスポンスヘッダーをPOSTメソッドにも追加
resource "aws_api_gateway_method_response" "time_management_post_method_response" {
    rest_api_id = aws_api_gateway_rest_api.time_management_api_gateway.id
    resource_id = aws_api_gateway_resource.time_management_api_resource.id
    http_method = "POST"
    status_code = "200"

    response_models = {
        "application/json" = "Empty"
    }

    response_parameters = {
        "method.response.header.Access-Control-Allow-Headers" = true
        "method.response.header.Access-Control-Allow-Methods" = true
        "method.response.header.Access-Control-Allow-Origin"  = true
    }
}

resource "aws_api_gateway_deployment" "time_management_deployment" {
    depends_on = [
        aws_api_gateway_integration.time_management_lambda_integration,
        aws_api_gateway_integration.time_management_options_integration
    ]

    rest_api_id = aws_api_gateway_rest_api.time_management_api_gateway.id
}

resource "aws_api_gateway_stage" "time_management_api_stage" {
    stage_name   = "prod"
    rest_api_id  = aws_api_gateway_rest_api.time_management_api_gateway.id
    deployment_id = aws_api_gateway_deployment.time_management_deployment.id
}

resource "aws_lambda_permission" "time_management_lambda" {
    statement_id  = "AllowAPIGatewayInvoke"
    action        = "lambda:InvokeFunction"
    function_name = aws_lambda_function.time_management_lambda_function.function_name
    principal     = "apigateway.amazonaws.com"

    source_arn = "${aws_api_gateway_rest_api.time_management_api_gateway.execution_arn}/prod/*/timemanagement"
}

resource "aws_api_gateway_rest_api_policy" "time_management_policy" {
    rest_api_id = aws_api_gateway_rest_api.time_management_api_gateway.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = "*",
                Action = "execute-api:Invoke",
                Resource = "${aws_api_gateway_rest_api.time_management_api_gateway.execution_arn}/*"
            }
        ],
    })
}