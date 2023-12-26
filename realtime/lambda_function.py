import json
import boto3
import requests
import os
from datetime import datetime


def generate_message(sentiment, user_input):
    sentiment_responses = {
        'Positive': '機嫌が良いようです',
        'Negative': '機嫌が悪いようです',
        'Neutral': '落ち着いているようです',
        'Irrelevant': '無関心なようです'
    }

    return f"ユーザーの入力: '{user_input}' - {sentiment_responses.get(sentiment, '不明な感情')}"


def translate_text(text, target_language):
    translate = boto3.client('translate')
    result = translate.translate_text(Text=text,
                                      SourceLanguageCode="ja",
                                      TargetLanguageCode=target_language)
    return result.get('TranslatedText')


def save_to_dynamodb(user_id, sentiment, openai_response):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('UserFeedback')
    response = table.put_item(
        Item={
            'UserID': user_id,
            'Timestamp': str(datetime.now()),
            'Sentiment': sentiment,
            'Response': openai_response
        }
    )

    return response


def lambda_handler(event, context):
    try:
        print('Received event:', event)

        slack_webhook_url = os.environ['SLACK_WEBHOOK_URL']
        openai_api_key = os.environ['OPENAI_API_KEY']
        print('Environment variables retrieved')

        if 'body' in event:
            body = json.loads(event['body'])
            user_input = body.get('input', '').strip()
            print('User input:', user_input)
        else:
            print('No body in the event')
            return {
                'statusCode': 400,
                'body': json.dumps('No body in the request')
            }

        if not user_input:
            print('User input is empty')
            return {
                'statusCode': 400,
                'body': json.dumps('Input text cannot be empty.')
            }

        translated_input = translate_text(user_input, "en")
        print('Translated input:', translated_input)

        message = f"ユーザーの入力: '{translated_input}'"
        print('Message to OpenAI:', message)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_api_key}'
        }

        data = {
            'prompt': message,
            'max_tokens': 150
        }

        openai_response = requests.post(
            'https://api.openai.com/v1/engines/davinci-codex/completions', headers=headers, json=data)
        openai_answer = openai_response.json().get('choices')[
            0].get('text').strip()
        print('OpenAI response:', openai_answer)

        slack_message = {'text': openai_answer}
        requests.post(slack_webhook_url, data=json.dumps(slack_message))
        print('Message sent to Slack')

        user_id = event.get('user_id', 'unknown')
        print('User ID:', user_id)

        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('UserFeedback')
        table.put_item(
            Item={
                'UserID': user_id,
                'Timestamp': str(datetime.now()),
                'Response': openai_answer
            }
        )
        print('Data saved to DynamoDB')

        return {
            'statusCode': 200,
            'body': json.dumps('Message sent to Slack and OpenAI successfully')
        }

    except Exception as e:
        print(f"Error processing the request: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps('Error processing the request')
        }
