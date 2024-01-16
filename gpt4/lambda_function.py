import json
import logging
import openai
import os
import requests

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
        user_input = body.get("message", "")

        logger.info(f"User Input: {user_input}")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        )

        ai_response = response.choices[0].message["content"]
        logger.info(f"AI Response: {ai_response}")

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
