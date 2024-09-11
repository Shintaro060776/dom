resource "aws_iam_role" "time_management_role" {
    name = "time_management"

    assume_role_policy = jsonencode({
        Version = "2012-10-17",
        Statement = [{
            Action = "sts:AssumeRole",
            Effect = "Allow",
            Principal = {
                Service = "lambda.amazonaws.com"
            },
        }]
    })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution_time_management" {
    role = aws_iam_role.time_management_role.name
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_dynamodb_access" {
    role = aws_iam_role.time_management_role.name
    policy_arn = "arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess"
}

# Lambda Layerの定義
resource "aws_lambda_layer_version" "time_management" {
    filename = "/home/runner/work/dom/dom/lambda_layer/lambda_layer.zip"
    layer_name = "time_management"
    compatible_runtimes = ["python3.11"]
    description = "Layer with OpenAI"
}

resource "aws_lambda_layer_version" "time_management2" {
    filename = "/home/runner/work/dom/dom/new_layer_dir/new_lambda_layer.zip"  # 新しく作成したLayerのZIPファイルのパス
    layer_name = "new_time_management_layer"
    compatible_runtimes = ["python3.11"]
    description = "Layer with OpenAI and AWS Lambda Powertools"
}

# Lambda関数の定義
resource "aws_lambda_function" "time_management_lambda_function" {
    function_name = "time_management"

    filename = "/home/runner/work/dom/dom/gpt4/lambda_function.zip"
    handler = "lambda_function.lambda_handler"
    role = aws_iam_role.time_management_role.arn
    runtime = "python3.11"

    timeout = 900

    # Layerの指定（layer_nameではなくlayersを使います）
    layers = [
        aws_lambda_layer_version.time_management.arn,
        aws_lambda_layer_version.time_management2.arn,
    ]

    environment {
        variables = {
            OPENAI_API_KEY = data.aws_ssm_parameter.openai_api_key.value
        }
    }
}

resource "aws_cloudwatch_log_group" "time_management_lambda_log_group" {
    name = "/aws/lambda/${aws_lambda_function.time_management_lambda_function.function_name}"
    retention_in_days = 14
}