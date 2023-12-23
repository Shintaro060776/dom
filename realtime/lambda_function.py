import json
import boto3
import requests
import os


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


def lambda_handler(event, context):
    try:
        slack_webhook_url = os.environ['SLACK_WEBHOOK_URL']
        openai_api_key = os.environ['OPENAI_API_KEY']

        sentiment = event['sentiment']
        user_input = event['user_input']

        translated_input = translate_text(user_input, "en")

        message = generate_message(sentiment, translated_input)

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
        slack_message = {'text': openai_answer}
        response = requests.post(
            slack_webhook_url, data=json.dumps(slack_message))

        return {
            'statusCode': 200,
            'body': json.dumps('Message sent to Slack and OpenAI successfully')
        }

    except Exception as e:
        print(e)
        return {
            'statusCode': 500,
            'body': json.dumps('Error processing the request')
        }
