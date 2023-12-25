resource "aws_api_gateway_rest_api" "realtime" {
    name = "realtime"
    description = "Realtime API"

    endpoint_configuration {
        types = ["EDGE"]
    }
}

resource "aws_api_gateway_resource" "realtime_resource" {
    rest_api_id = aws_api_gateway_rest_api.realtime.id
    parent_id   = aws_api_gateway_rest_api.realtime.root_resource_id
    path_part   = "realtime"  
}

resource "aws_api_gateway_method" "realtime_method" {
    rest_api_id = aws_api_gateway_rest_api.realtime.id
    resource_id = aws_api_gateway_resource.realtime_resource.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration_realtime" {
    rest_api_id = aws_api_gateway_rest_api.realtime.id
    resource_id = aws_api_gateway_resource.realtime_resource.id
    http_method = aws_api_gateway_method.realtime_method.http_method

    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.lambda_function.invoke_arn
}

resource "aws_api_gateway_deployment" "realtime_deployment" {
    depends_on = [aws_api_gateway_integration.lambda_integration]
    rest_api_id = aws_api_gateway_rest_api.realtime.id
    stage_name = "prod"
}

resource "aws_api_gateway_stage" "realtime_stage" {
    deployment_id = aws_api_gateway_deployment.realtime_deployment.id
    rest_api_id = aws_api_gateway_rest_api.realtime.id
    stage_name = "prod"

    access_log_settings {
        destination_arn = aws_cloudwatch_log_group.api_gw_log_group.arn
        format = "{\"requestId\":\"$context.requestId\", \"ip\":\"$context.identity.sourceIp\", \"requestTime\":\"$context.requestTime\", \"httpMethod\":\"$context.httpMethod\", \"status\":\"$context.status\", \"protocol\":\"$context.protocol\", \"responseLength\":\"$context.responseLength\"}"
    }
}

resource "aws_cloudwatch_log_group" "api_gw_log_group" {
    name = "/aws/apigateway/realtime"
    retention_in_days = 7
}

resource "aws_api_gateway_account" "main" {
    cloudwatch_role_arn = aws_iam_role.api_gw_cloudwatch.arn
}

resource "aws_iam_role" "api_gw_cloudwatch" {
    name = "api-gw-cloudwatch-role"

    assume_role_policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Principal = {
                    Service = "apigateway.amazonaws.com"
                },
                Action = "sts:AssumeRole"
            },
        ],
    })
}

resource "aws_iam_role_policy" "api_gw_cloudwatch" {
    name = "api-gw-cloudwatch-policy"
    role = aws_iam_role.api_gw_cloudwatch.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Action = [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams",
                    "logs:PutLogEvents",
                    "logs:GetLogEvents",
                    "logs:FilterLogEvents"
                ],
                Resource = "*"
            },
        ],
    })
}

resource "aws_lambda_permission" "api_gateway_permission" {
    statement_id = "AllowExecutionFromAPIGateway"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.lambda_function.function_name
    principal = "apigateway.amazonaws.com"
    source_arn = "${aws_api_gateway_rest_api.realtime.execution_arn}/*/*/realtime"
}