import json
import logging
import openai
import os
import boto3
import requests

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def send_slack_notification(user_input, ai_response):
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    message = f"User Input: {user_input}\nAI Response: {ai_response}"
    slack_data = {'text': message}
    response = requests.post(webhook_url, json=slack_data)
    return response


def save_summary_to_dynamodb(file_key, summary):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('SummaryTable')
    table.put_item(
        Item={
            'file_key': file_key,
            'summary': summary
        }
    )


def lambda_handler(event, context):
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']

        s3_client = boto3.client('s3')
        audio_file = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        audio_content = audio_file['Body'].read()

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_content,
            response_format="text"
        )

        text = transcript['text']
        logger.info(f"Transcript: {text}")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ]
        )

        ai_response = response.choices[0].message.content
        logger.info(f"AI Response: {ai_response}")

        save_summary_to_dynamodb(file_key, ai_response)

        send_slack_notification(text, ai_response)

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
