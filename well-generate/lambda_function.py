import boto3
import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

translate_client = boto3.client('translate')
sagemaker_runtime = boto3.client('sagemaker-runtime')


def lambda_handler(event, context):
    category_endpoint = os.environ['CATEGORY_CLASSIFICATION_ENDPOINT']
    text_generation_endpoint = os.environ['TEXT_GENERATION_ENDPOINT']

    try:
        logger.info(f"Received event: {event}")

        if event.get("body"):
            input_text = json.loads(event["body"]).get("text")
            logger.info(f"Received input text: {input_text}")
        else:
            return error_response(400, "No text provided in request body")

        translated_text = translate_text(input_text, target_language='en')
        logger.info(
            f"Translated text for category classification: {translated_text}")

        category_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=category_endpoint,
            ContentType='text/plain',
            Body=translated_text
        )
        category_result = category_response['Body'].read().decode()
        logger.info(f"Category classification response: {category_response}")
        logger.info(f"Category classification result: {category_result}")

        text_generation_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=text_generation_endpoint,
            ContentType='application/json',
            Body=json.dumps(
                {'category': category_result, 'text': translated_text})
        )
        generated_text = text_generation_response['Body'].read().decode()
        logger.info(f"Generated text: {generated_text}")

        translated_generated_text = translate_text(
            generated_text, target_language='ja')
        logger.info(f"Translated generated text: {translated_generated_text}")

        return {
            'statusCode': 200,
            'body': json.dumps({'generatedText': translated_generated_text})
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return error_response(500, "Internal server error: " + str(e))


def translate_text(text, target_language='ja'):
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
