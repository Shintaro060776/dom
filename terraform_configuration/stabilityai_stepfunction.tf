resource "aws_iam_role" "step_function_role" {
    assume_role_policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = "sts:AssumeRole",
                Effect = "Allow",
                Principal = {
                    Service = "states.amazonaws.com"
                }
            }
        ]
    })
}

resource "aws_iam_policy" "lambda_invoke_stepfunction_policy" {
    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [
                    "lambda:InvokeFunction"
                ],
                Effect = "Allow",
                Resource = [
                    aws_lambda_function.stabilityai2.arn,
                    aws_lambda_function.stabilityai3.arn
                ]
            }
        ]
    })
}

resource "aws_iam_policy" "cloudwatch_logs_policy" {
    name = "CloudWatchLogsPolicyForStepFunctions"
    description = "Allow Step Functions to write logs to CloudWatch"

    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Effect = "Allow",
                Action = [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                Resource = "arn:aws:logs:*:*:*"
            }
        ]
    })
}

resource "aws_iam_role_policy_attachment" "cloudwatch_logs_policy_attachment" {
    role = aws_iam_role.step_function_role.name
    policy_arn = aws_iam_policy.cloudwatch_logs_policy.arn
}

resource "aws_iam_role_policy_attachment" "step_functions_lambda_attachment" {
    role = aws_iam_role.step_function_role.name
    policy_arn = aws_iam_policy.lambda_invoke_stepfunction_policy.arn
}

resource "aws_cloudwatch_log_group" "step_functions_log_group" {
    name = "/aws/vendedlogs/states/VideoGenerationStateMachine"
    retention_in_days = 30
}

resource "aws_sfn_state_machine" "video_generation_state_machine" {
    name = "VideoGenerationStateMachine"
    role_arn = aws_iam_role.step_function_role.arn

    logging_configuration {
        include_execution_data = true
        level = "ALL"
        destinations {
            cloud_watch_logs_log_group {
                log_group_arn = aws_cloudwatch_log_group.step_functions_log_group.arn
            }
        }
    }

    definition = <<EOF
    {
        "Comment": "Video Generation State machine",
        "StartAt": "StartVideoGeneration",
        "States": {
            "StartVideoGeneration": {
                "Type": "Task",
                "Resource": "${aws_lambda_function.stabilityai2.arn}",
                "Next": "CheckGenerationStatus"
            },
            "CheckGenerationStatus": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.statusCode",
                        "NumericEquals": 200,
                        "Next": "ProcessVideo"
                    },
                    {
                        "Variable": "$.statusCode",
                        "NumericEquals": 202,
                        "Next": "WaitForCompletion"
                    }
                ],
                "Default": "GenerationFailed"
            },
            "ProcessVideo": {
                "Type": "Task",
                "Resource": "${aws_lambda_function.stabilityai3.arn}",
                "End": true
            },
            "WaitForCompletion": {
                "Type": "Wait",
                "Seconds": 30,
                "Next": "StartVideoGeneration"
            },
            "GenerationFailed": {
                "Type": "Fail",
                "Cause": "Video Generation Failed",
                "Error": "Status Code Not 200"
            }
        }
    }
    EOF
}