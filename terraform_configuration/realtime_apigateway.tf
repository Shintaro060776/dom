resource "aws_api_gateway_rest_api" "realtime" {
    name = "realtime"
    description = "API Gateway for my lambda function"
}

resource "aws_api_gateway_resource" "realtime_resource" {
    rest_api_id = aws_api_gateway_rest_api.realtime.id
    parent_id = aws_api_gateway_rest_api.realtime.root_resource_id
    path_part = "realtime"
}

resource "aws_api_gateway_method" "my_api_method_realtime" {
  rest_api_id   = aws_api_gateway_rest_api.realtime.id
  resource_id   = aws_api_gateway_resource.realtime_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration_realtime" {
  rest_api_id = aws_api_gateway_rest_api.realtime.id
  resource_id = aws_api_gateway_resource.realtime_resource.id
  http_method = aws_api_gateway_method.my_api_method_realtime.http_method

  integration_http_method = "POST"
  type                    = "AWS"
  uri                     = aws_lambda_function.lambda_function.invoke_arn

  request_templates = {
    "application/json" = jsonencode({
        "statusCode" = 200
    })
  }
}

resource "aws_api_gateway_integration_response" "realtime_integration_response_200" {
    depends_on = [aws_api_gateway_integration.lambda_integration_realtime]

    rest_api_id = aws_api_gateway_rest_api.realtime.id
    resource_id = aws_api_gateway_resource.realtime_resource.id
    http_method = aws_api_gateway_method.my_api_method_realtime.http_method
    status_code = "200"
}

resource "aws_api_gateway_deployment" "realtime_deployment" {
    depends_on = [
        aws_api_gateway_integration.lambda_integration_realtime
    ]

    rest_api_id = aws_api_gateway_rest_api.realtime.id
    stage_name = "prod"
}


resource "aws_lambda_permission" "api_gateway_invoke_realtime" {
    statement_id = "AllowExecutionFromAPIGateway"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.lambda_function.function_name
    principal = "apigateway.amazonaws.com"
    source_arn = "${aws_api_gateway_rest_api.realtime.execution_arn}/*/*/*"
}

resource "aws_api_gateway_method_settings" "settings_realtime" {
  rest_api_id = aws_api_gateway_rest_api.realtime.id
  stage_name  = aws_api_gateway_stage.realtime.stage_name
  method_path = "*/*"

  settings {
    logging_level       = "INFO" 
    metrics_enabled     = true
    data_trace_enabled  = true  
  }
}

resource "aws_api_gateway_stage" "realtime" {
    stage_name = "prod"
    rest_api_id = aws_api_gateway_rest_api.realtime.id
    deployment_id = aws_api_gateway_deployment.realtime_deployment.id

    access_log_settings {
        destination_arn = aws_cloudwatch_log_group.realtime.arn
        format = "{\"requestId\":\"$context.requestId\", \"ip\": \"$context.identity.sourceIp\", \"caller\":\"$context.identity.caller\", \"user\":\"$context.identity.user\", \"requestTime\":\"$context.requestTime\", \"httpMethod\":\"$context.httpMethod\", \"resourcePath\":\"$context.resourcePath\", \"status\":\"$context.status\", \"protocol\":\"$context.protocol\", \"responseLength\":\"$context.responseLength\"}"
    }

    xray_tracing_enabled = true
}

resource "aws_cloudwatch_log_group" "realtime" {
    name = "/aws/api-gateway/realtime"
}

resource "aws_iam_role" "api_gateway_cloudwatch_role_realtime" {
    name = "api_gateway_cloudwatch_role_realtime"

    assume_role_policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = "sts:AssumeRole",
                Effect = "Allow",
                Principal = {
                    Service = "apigateway.amazonaws.com"
                },
            },
        ],
    })
}

resource "aws_iam_role_policy" "api_gateway_cloudwatch_policy_realtime" {
    name = "api_gateway_cloudwatch_policy_realtime"
    role = aws_iam_role.api_gateway_cloudwatch_role_realtime.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                Resource = "arn:aws:logs:*:*:*",
                Effect = "Allow",
            },
        ],
    })
}
