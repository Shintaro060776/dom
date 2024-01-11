resource "aws_api_gateway_rest_api" "claim_api" {
    name = "claim-api-unique"
    description = "API for handling user claims"

    endpoint_configuration {
        type = ["REGIONAL"]
    }
}

resource "aws_api_gateway_resource" "api_resource" {
    rest_api_id = aws_api_gateway_rest_api.clamin_api.id
    parent_id = aws_api_gateway_rest_api.claim_api.root_resource_id
    path_part = "claim_handler"
}

resource "aws_api_gateway_method" "api_method" {
    rest_api_id = aws_api_gateway_rest_api.claim_api.id
    resource_id = aws_api_gateway_resource.api_resource.id
    http_method = "POST"
    authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration_claim" {
    rest_api_id = aws_api_gateway_rest_api.claim_api.id
    resource_id = aws_api_gateway_resource.api_resource.id
    http_method = aws_api_gateway_method.api_method.http_method
    integration_http_method = "POST"
    type = "AWS_PROXY"
    uri = aws_lambda_function.claim_handler_lambda.invoke_arn
}

resource "aws_lambda_permission" "api_gateway_permission" {
    statement_id = "AllowAPIGatewayInvoke"
    action = "lambda:InvokeFunction"
    function_name = aws_lambda_function.claim_handler_lambda.function_name
    principal = "apigateway.amazonaws.com"
    source_arn = "${aws_api_gateway_rest_api.claim_api.execution_arn}/*/*"
}


resource "aws_api_gateway_account" "api_gw_account" {
    cloudwatch_role_arn = aws_iam_role.api_gateway_cloudwatch.arn
}

resource "aws_iam_role" "api_gateway_cloudwatch" {
    name = "api-gateway-cloudwatch-role-unique"

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

resource "aws_iam_role_policy" "api_gateway_cloudwatch_policy_claim" {
    name = "api-gateway-cloudwatch-policy-unique"
    role = aws_iam_role.api_gateway_cloudwatch.id

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStream"
                ],
                Effect = "Allow",
                Resource = "*"
            },
        ],
    })
}

resource "aws_api_gateway_deployment" "claim_api_deployment" {
    rest_api_id = aws_api_gateway_rest_api.claim_api.id
    stage_name = "prod"

    depends_on = [
        aws_api_gateway_integration.lambda_integration_claim
    ]
}

resource "aws_api_gateway_stage" "api_stage" {
    stage_name = "prod"
    rest_api_id = aws_api_gateway_rest_api.claim_api.id
    deployment_id = aws_api_gateway_deployment.claim_api_deployment.id

    access_log_settings {
        destination_arn = aws_cloudwatch_log_group.api_gw_log_group.arn
        format = "{'requestId':'$context.requestId','ip':'$context.identity.sourceIp','requestTime':'$context.requestTime','httpMethod':'$context.httpMethod','status':'$context.status','protocol':'$context.protocol','responseLength':'$context.responseLength'}"
    }
}

resource "aws_cloudwatch_log_group" "api_gw_log_group" {
    name = "/aws/apigateway/claim-api-unique"
}