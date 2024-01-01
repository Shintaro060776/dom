resource "aws_api_gateway_rest_api" "generate_api" {
    name = "GenerateAPI"
    description = "Rest API for text classification and generation"
}

resource "aws_api_gateway_resource" "generate_resource" {
    rest_api_id = aws_api_gateway_rest_api.generate_api.id
    parent_id = aws_api_gateway_rest_api.generate_api.root_resource_id
    path_part = "generate"
}

resource "aws_api_gateway_method" "generate_method" {
    rest_api_id = aws_api_gateway_rest_api.generate_api.id
    resource_id = aws_api_gateway_resource.generate_resource.id
    http_method = "GET"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "generate_lambda_integration" {
    rest_api_id = aws_api_gateway_rest_api.generate_api.id
    resource_id = aws_api_gateway_resource.generate_resource.id
    http_method = aws_api_gateway_method.generate_method.http_method
    type = "AWS_PROXY"

    uri = aws_lambda_function.generate_lambda.invoke_arn
    integration_http_method = "POST"
}

resource "aws_api_gateway_deployment" "generate_deployment" {
    depends_on = [
        aws_api_gateway_integration.generate_lambda_integration
    ]

    rest_api_id = aws_api_gateway_rest_api.generate_api.id
    stage_name = "prod"
}

resource "aws_lambda_permission" "generate_api_gateway_invoke" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.generate_lambda.function_name
    principal = "apigateway.amazonaws.com"
    source_arn = "${aws_api_gateway_rest_api.generate_api.execution_arn}/*/*/*"
}

resource "aws_cloudwatch_log_group" "generate_log_group" {
    name = "/aws/apigateway/GenerateAPI"

    retention_in_days = 90
}

resource "aws_api_gateway_stage" "generate" {
    stage_name = "prod"
    rest_api_id = aws_api_gateway_rest_api.generate_api.id
    deployment_id = aws_api_gateway_deployment.generate_deployment.id

    xray_tracing_enabled = true

    access_log_settings {
        description_arn = aws_cloudwatch_log_group.generate.generate_log_group.arn
        format = jsonencode({
            request_id       = "$context.requestId",
            ip               = "$context.identity.sourceIp",
            caller           = "$context.identity.caller",
            user             = "$context.identity.user",
            request_time     = "$context.requestTime",
            http_method      = "$context.httpMethod",
            status           = "$context.status",
            protocol         = "$context.protocol",
            response_length  = "$context.responseLength"
        })
    }
}

resource "aws_iam_role_policy" "api_gateway_cloudwatch_log_policy" {
    name = "api_gateway_cloudwatch_log_policy"
    role = aws_iam_role.api_gateway_lambda_role.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            Effect = "Allow",
            Action = "logs:CreateLogGroup",
            Resource = "arn:aws:logs:ap-northeast-1:715573459931:*"
        ],
        {
            Effect = "Allow",
            Action = [
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            Resource = [
                "${aws_cloudwatch_log_group.generate_log_group.arn}:*"
            ]
        }
    })
}