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


resource "aws_iam_role_policy_attachment" "step_functions_lambda_attachment" {
    role = aws_iam_role.step_function_role.name
    policy_arn = aws_iam_policy.lambda_invoke_stepfunction_policy.arn
}


resource "aws_sfn_state_machine" "video_generation_state_machine" {
    name = "VideoGenerationStateMachine"
    role_arn = aws_iam_role.step_function_role.arn

    definition = <<EOF
{
  "Comment": "Video Generation State machine",
  "StartAt": "StartVideoGeneration",
  "States": {
    "StartVideoGeneration": {
      "Type": "Task",
      "Resource": "${aws_lambda_function.stabilityai2.arn}",
      "Next": "CheckVideoStatus"
    },
    "CheckVideoStatus": {
        "Type": "Task",
        "Resource": "${aws_lambda_function.stabilityai3.arn}",
        "InputPath": "$.body",
        "Next": "CheckGenerationStatus"
    },
    "CheckGenerationStatus": {
        "Type": "Choice",
        "Choices": [
            {
                "Variable": "$.videoStatus.statusCode",
                "NumericEquals": 200,
                "Next": "ProcessVideo"
            },
            {
                "Variable": "$.videoStatus.statusCode",
                "NumericEquals": 202,
                "Next": "WaitForCompletion"
            }
        ],
        "Default": "GenerationFailed"
    },
    "WaitForCompletion": {
      "Type": "Wait",
      "Seconds": 5,
      "Next": "CheckGenerationStatus"
    },
    "ProcessVideo": {
      "Type": "Task",
      "Resource": "${aws_lambda_function.stabilityai3.arn}",
      "End": true
    },
    "GenerationFailed": {
      "Type": "Fail",
      "Cause": "Video Generation Failed",
      "Error": "Status Code Not 200"
    }
  }
}
    EOF

    #     logging_configuration {
    #     log_destination        = aws_cloudwatch_log_group.step_functions_log_group.arn
    #     include_execution_data = true
    #     level                  = "ERROR"
    # }
}