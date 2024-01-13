import json
import boto3
import os
import logging
from botocore.exceptions import ClientError
import requests

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def send_to_slack(webhook_url, message):
    response = requests.post(webhook_url, json={"text": message})
    if response.status_code != 200:
        raise ValueError(
            f"Request to Slack returned an error {response.status_code}, the response is:\n{response.text}")


def lambda_handler(event, context):
    sentiment_endpoint = os.environ['SAGEMAKER_ENDPOINT_SENTIMENT']
    text_gen_endpoint = os.environ['SAGEMAKER_ENDPOINT_TEXT_GEN']
    slack_webhook_url = os.environ['SLACK_WEBHOOK_URL']
    dynamodb_table_name = os.environ['DYNAMODB_TABLE_NAME']

    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing request body: {e}")
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid request body"})}

    user_input = body.get("user_input", "")
    logger.info(f"User input: {user_input}")

    if not user_input:
        return {"statusCode": 400, "body": json.dumps({"error": "User input is required"})}

    try:
        translate_client = boto3.client('translate')
        translated_text = translate_client.translate_text(
            Text=user_input,
            SourceLanguageCode='ja',
            TargetLanguageCode='en'
        ).get('TranslatedText')
        logger.info(f"Translated text: {translated_text}")

        sagemaker_runtime = boto3.client('sagemaker-runtime')
        sentiment_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=sentiment_endpoint,
            ContentType='text/plain',
            Body=translated_text
        )
        sentiment_result = json.loads(
            sentiment_response['Body'].read().decode())
        logger.info(f"Sentiment analysis result: {sentiment_result}")

        text_gen_input = {"sentiment": sentiment_result,
                          "text": translated_text}
        text_gen_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=text_gen_endpoint,
            ContentType='application/json',
            Body=json.dumps(text_gen_input)
        )
        text_gen_result = json.loads(text_gen_response['Body'].read().decode())
        logger.info(f"Text generation result: {text_gen_result}")

        final_text = translate_client.translate_text(
            Text=text_gen_result['generated_text'],
            SourceLanguageCode='en',
            TargetLanguageCode='ja'
        ).get('TranslatedText')
        logger.info(f"Final text: {final_text}")

        send_to_slack(slack_webhook_url, f'新しいクレーム対応: {final_text}')
        # slack_client = boto3.client('lambda')
        # slack_message = {'text': f'新しいクレーム対応: {final_text}'}
        # slack_client.invoke(
        #     FunctionName=slack_webhook_url,
        #     InvocationType='Event',
        #     Payload=json.dumps(slack_message)
        # )

        dynamodb_client = boto3.client('dynamodb')
        dynamodb_client.put_item(
            TableName=dynamodb_table_name,
            Item={
                'UserInput': {'S': user_input},
                'TranslatedText': {'S': translated_text},
                'SentimentResult': {'S': json.dumps(sentiment_result)},
                'GeneratedText': {'S': text_gen_result},
                'FinalText': {'S': final_text}
            }
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "original_text": user_input,
                "translated_text": translated_text,
                "sentiment": sentiment_result,
                "generated_text": text_gen_result,
                "final_text": final_text
            })
        }

    except ClientError as e:
        logger.error(f"An error occurred: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
