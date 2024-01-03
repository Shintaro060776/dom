import boto3
import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

translate_client = boto3.client('translate')


def lambda_handler(event, context):
    sagemaker_runtime = boto3.client('sagemaker-runtime')

    category_endpoint = os.environ['CATEGORY_CLASSIFICATION_ENDPOINT']
    text_generation_endpoint = os.environ['TEXT_GENERATION_ENDPOINT']

    try:
        input_text = event["body"]
        logger.info(f"Received input: {input_text}")

        translated_text = translate_text(input_text)
        logger.info(f"Translated text: {translated_text}")

        category_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=category_endpoint,
            ContentType='text/plain',
            Body=translated_text
        )
        category_result = category_response['Body'].read().decode()
        logger.info(f"Category classification result: {category_result}")

        text_generation_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=text_generation_endpoint,
            ContentType='application/json',
            Body=json.dumps(
                {'category': category_result, 'text': translated_text})
        )
        generated_text = text_generation_response['Body'].read().decode()
        logger.info(f"Generated text: {generated_text}")

        return {
            'statusCode': 200,
            'body': json.dumps({'generatedText': generated_text})
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return error_response(500, "Internal server error: " + str(e))


def translate_text(text, target_language='en'):
    try:
        response = translate_client.translate_text(
            Text=text,
            SourceLanguageCode='auto',
            TargetLanguageCode=target_language
        )
        return response.get('TranslatedText', text)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text


def error_response(status_code, message):
    return {
        'statusCode': status_code,
        'body': json.dumps({'error': message})
    }
