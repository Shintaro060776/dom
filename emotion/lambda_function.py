import json
import boto3
import os
import requests

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SAGEMAKER_ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL')


def lambda_handler(event, context):
    translate = boto3.client('translate')
    sagemaker_runtime = boto3.client('runtime.sagemaker')

    input_text = event['body']

    translated_text = translate.translate_text(
        Text=input_text,
        SourceLanguageCode='ja',
        TargetLanguageCode='en'
    )['TranslatedText']

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType='text/plain',
        Body=translated_text
    )
    sentiment = json.loads(response['Body'].read().decode())

    response_text = input_text + \
        (" ユーザーは機嫌が良いようです。" if sentiment == "positive" else " ユーザーは機嫌が悪いようです。")

    url = "https://api.openai.com/v1/engines/gpt-3.5-turbo/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "prompt": response_text,
        "max_tokens": 150,
        "temperature": 0.1,
        "top_p": 1.0
    }

    openai_response = requests.post(url, headers=headers, json=payload)

    if openai_response.status_code == 200:
        openai_result = openai_response.json()
        text_to_send = openai_result.get('choices', [{}])[0].get('text', '')

        slack_message = {
            "text": f"OpenAI Response: {text_to_send}"
        }
        slack_response = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(slack_message),
            headers={'Content-Type': 'application/json'}
        )

        if slack_response.status_code != 200:
            return {
                'statusCode': slack_response.status_code,
                'body': json.dumps({"error": "Failed to send message to Slack"})
            }

        return {
            'statusCode': 200,
            'body': json.dumps(openai_result)
        }
    else:
        return {
            'statusCode': openai_response.status_code,
            'body': json.dumps({"error": "OpenAI API request failed"})
        }
