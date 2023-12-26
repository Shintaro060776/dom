import json
import boto3
import requests
import os
from datetime import datetime

sagemaker_runtime = boto3.client('sagemaker-runtime')


def generate_message(sentiment, user_input):
    sentiment_responses = {
        'Positive': '機嫌が良いようです',
        'Negative': '機嫌が悪いようです',
        'Neutral': '落ち着いているようです',
        'Mixed': '感情が混ざっているようです'
    }
    return f"ユーザーの入力: '{user_input}' - 感情分析結果: {sentiment_responses.get(sentiment, '不明な感情')}"


def translate_text(text, target_language):
    translate = boto3.client('translate')
    result = translate.translate_text(Text=text,
                                      SourceLanguageCode="ja",
                                      TargetLanguageCode=target_language)
    return result.get('TranslatedText')


def get_sentiment_from_sagemaker(translated_text):
    try:
        print(f'Sending to SageMaker: {translated_text}')
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=os.environ['SAGEMAKER_ENDPOINT_NAME'],
            ContentType='text/plain',
            Body=translated_text
        )
        response_body = response['Body'].read().decode()
        print(f'Received from SageMaker: {response_body}')
        if not response_body:
            raise ValueError("Empty response from SageMaker")

        return response_body
    except Exception as e:
        print(f"SageMaker invocation failed: {str(e)}")
        raise


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
        sentiment = get_sentiment_from_sagemaker(translated_input)

        message = generate_message(sentiment, user_input)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_api_key}'
        }

        data = {
            'prompt': message,
            'max_tokens': 150
        }

        print('Sending to OpenAI:', json.dumps(data))
        openai_response = requests.post(
            'https://api.openai.com/v1/engines/davinci/completions',
            headers=headers,
            json=data
        )

        try:
            openai_response_json = openai_response.json()

            print('Received from OpenAI:', json.dumps(openai_response_json))

            if 'choices' in openai_response_json:
                openai_answer = openai_response_json.get(
                    'choices')[0].get('text').strip()
            else:
                raise ValueError("Invalid response from openAI")

            slack_message = {'text': openai_answer}
            requests.post(slack_webhook_url, data=json.dumps(slack_message))
            print('Message sent to Slack')
        except Exception as e:
            print(f"Error processing the request: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps('Error processing the request')
            }

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
            'body': json.dumps({'message': openai_answer})
        }

    except Exception as e:
        print(f"Error processing the request: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps('Error processing the request')
        }
