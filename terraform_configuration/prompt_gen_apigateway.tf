# resource "aws_api_gateway_rest_api" "prompt_gen" {
#     name = "prompt_gen"
#     description = "API for generating images with StabilityAI"
# }

# resource "aws_api_gateway_resource" "prompt_gen" {
#     rest_api_id = aws_api_gateway_rest_api.prompt_gen.id
#     parent_id = aws_api_gateway_rest_api.prompt_gen.root_resource_id
#     path_part = "generate"
# }

# resource "aws_api_gateway_method" "prompt_gen" {
#     rest_api_id = aws_api_gateway_rest_api.prompt_gen.id
#     resource_id = aws_api_gateway_resource.prompt_gen.id
#     http_method = "POST"
#     authorization = "NONE"
# }

# resource "aws_api_gateway_integration" "prompt_gen" {
#     rest_api_id = aws_api_gateway_rest_api.prompt_gen.id
#     resource_id = aws_api_gateway_resource.prompt_gen.id
#     http_method = aws_api_gateway_method.prompt_gen.http_method

#     integration_http_method = "POST"
#     type = "AWS_PROXY"
#     uri = aws_lambda_function.prompt_gen.invoke_arn
# }

# resource "aws_api_gateway_deployment" "prompt_gen" {
#     depends_on = [
#         aws_api_gateway_integration.prompt_gen
#     ]

#     rest_api_id = aws_api_gateway_rest_api.prompt_gen.id
#     description = "Initial deployment"
# }

# resource "aws_api_gateway_stage" "prompt_gen" {
#     deployment_id = aws_api_gateway_deployment.prompt_gen.id
#     rest_api_id = aws_api_gateway_rest_api.prompt_gen.id
#     stage_name = "prod"
# }

# resource "aws_lambda_permission" "prompt_gen" {
#     statement_id = "AllowAPIGatewayInvoke"
#     action = "lambda:InvokeFunction"
#     function_name = aws_lambda_function.prompt_gen.function_name
#     principal = "apigateway.amazonaws.com"

#     source_arn = "${aws_api_gateway_rest_api.prompt_gen.execution_arn}/prod/*/generate"
# }

# resource "aws_api_gateway_rest_api_policy" "prompt_gen" {
#     rest_api_id = aws_api_gateway_rest_api.prompt_gen.id

#     policy = jsonencode({
#         Version = "2012-10-17",
#         Statement = [
#             {
#                 Effect = "Allow",
#                 Principal = "*",
#                 Action = "execute-api:Invoke",
#                 Resource = "${aws_api_gateway_rest_api.prompt_gen.execution_arn}/*"
#             }
#         ],
#     })
# }