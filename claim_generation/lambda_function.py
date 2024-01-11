import json
import boto3
import os
from botocore.exceptions import ClientError


def lambda_handler(event, context):
    sentiment_endpoint = os.environ['SAGEMAKER_ENDPOINT_SENTIMENT']
    text_gen_endpoint = os.environ['SAGEMAKER_ENDPOINT_TEXT_GEN']
    slack_webhook_url = os.environ['SLACK_WEBHOOK_URL']
    dynamodb_table_name = os.environ['DYNAMODB_TABLE_NAME']

    user_input = event.get("user_input", "")

    translate_client = boto3.client('translate')
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    slack_client = boto3.client('lambda')
    dynamodb_client = boto3.client('dynamodb')

    try:
        translated_text = translate_client.translate_text(
            Text=user_input,
            SourceLanguageCode='ja',
            TargetLanguageCode='en'
        ).get('TranslatedText')

        sentiment_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=sentiment_endpoint,
            ContentType='text/plain',
            Body=translated_text
        )
        sentiment_result = json.loads(
            sentiment_response['Body'].read().decode())

        text_gen_input = {"sentiment": sentiment_result,
                          "text": translated_text}
        text_gen_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=text_gen_endpoint,
            ContentType='application/json',
            Body=json.dumps(text_gen_input)
        )
        text_gen_result = json.loads(text_gen_response['Body'].read().decode())

        final_text = translate_client.translate_text(
            Text=text_gen_result,
            SourceLanguageCode='en',
            TargetLanguageCode='ja'
        ).get('TranslatedText')

        slack_message = {
            'text': f'新しいクレーム対応: {final_text}'
        }
        slack_client.invoke(
            FunctionName=slack_webhook_url,
            InvocationType='Event',
            Payload=json.dumps(slack_message)
        )

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
        print(f"An error occurred: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
