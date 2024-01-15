import json
import openai
import os
import requests
from aws_lambda_powertools import Logger, Tracer

logger = Logger()
tracer = Tracer()


def send_slack_notification(user_input, ai_response):
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    message = f"User Input: {user_input}\nAI Response: {ai_response}"
    slack_data = {'text': message}
    response = requests.post(webhook_url, json=slack_data)
    return response


def lambda_handler(event, context):
    try:
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        body = json.loads(event["body"])
        user_input = body["user_input"]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        )

        ai_response = response["choices"][0]["message"]["content"]

        send_slack_notification(user_input, ai_response)

        return {
            "statusCode": 200,
            "body": json.dumps({"response": ai_response})
        }

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")

        send_slack_notification("Error occurred", str(e))

        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal Server Error"})
        }
