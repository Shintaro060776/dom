resource "aws_api_gateway_rest_api" "my_api" {
    name = "MyAPI"
    description = "API Gateway for my lambda function"
}

resource "aws_api_gateway_resource" "my_api_resource" {
    rest_api_id = aws_api_gateway_rest_api.my_api.id
    parent_id = aws_api_gateway_rest_api.my_api.root_resource_id
    path_part = "myresource"
}

resource "aws_api_gateway_method" "my_api_method" {
  rest_api_id   = aws_api_gateway_rest_api.my_api.id
  resource_id   = aws_api_gateway_resource.my_api_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.my_api.id
  resource_id = aws_api_gateway_resource.my_api_resource.id
  http_method = aws_api_gateway_method.my_api_method.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.my_lambda.invoke_arn
}

resource "aws_api_gateway_deployment" "my_api_deployment" {
    depends_on = [
        aws_api_gateway_integration.lambda_integration
    ]

    rest_api_id = aws_api_gateway_rest_api.my_api.id
    stage_name = "prod"
}


resource "aws_lambda_permission" "api_gateway_invoke" {
    statement_id = "AllowExecutionFromAPIGateway"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.my_lambda.function_name
    principal = "apigateway.amazonaws.com"
    source_arn = "${aws_api_gateway_rest_api.my_api.execution_arn}/*/*/*"
}

resource "aws_api_gateway_method_settings" "settings" {
    rest_api_id = aws_api_gateway_rest_api.my_api.id
    stage_name = aws_api_gateway_stage.example.stage_name
    method_path = "${aws_api_gateway_resource.my_api_resource.path_part}/POST"

    settings {
        logging_level = "ERROR"
        metrics_enabled = true
    }
}

resource "aws_api_gateway_stage" "example" {
    stage_name = "prod"
    rest_api_id = aws_api_gateway_rest_api.my_api.id
    deployment_id = aws_api_gateway_deployment.my_api_deployment.id

    access_log_settings {
        destination_arn = aws_cloudwatch_log_group.example.arn
        format = "{\"requestId\":\"$context.requestId\", \"ip\": \"$context.identity.sourceIp\", \"caller\":\"$context.identity.caller\", \"user\":\"$context.identity.user\", \"requestTime\":\"$context.requestTime\", \"httpMethod\":\"$context.httpMethod\", \"resourcePath\":\"$context.resourcePath\", \"status\":\"$context.status\", \"protocol\":\"$context.protocol\", \"responseLength\":\"$context.responseLength\"}"
    }

    xray_tracing_enabled = true
}

resource "aws_cloudwatch_log_group" "example" {
    name = "/aws/api-gateway/my-api"
}

resource "aws_iam_role" "api_gateway_cloudwatch_role" {
    name = "api_gateway_cloudwatch_role"

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

resource "aws_iam_role_policy" "api_gateway_cloudwatch_policy" {
    name = "api_gateway_cloudwatch_policy"
    role = aws_iam_role.api_gateway_cloudwatch_role.id

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
