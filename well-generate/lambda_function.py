import boto3
import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    sagemaker_runtime = boto3.client('sagemaker-runtime')

    category_endpoint = os.environ['CATEGORY_CLASSIFICATION_ENDPOINT']
    text_generation_endpoint = os.environ['TEXT_GENERATION_ENDPOINT']

    try:
        input_text = event["body"]
        logger.info(f"Received input: {input_text}")

        category_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=category_endpoint,
            ContentType='text/plain',
            Body=input_text
        )
        category_result = category_response['Body'].read().decode()
        logger.info(f"Category classification result: {category_result}")

        text_generation_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=text_generation_endpoint,
            ContentType='application/json',
            Body=json.dumps({'category': category_result})
        )
        generated_text = text_generation_response['Body'].read().decode()
        logger.info(f"Generated text: {generated_text}")

        return {
            'statusCode': 200,
            'body': json.dumps({'generatedText': generated_text})
        }

    except sagemaker_runtime.exceptions.ValidationError as e:
        logger.error(f"Validation error in SageMaker request: {e}")
        return error_response(400, "Validation error in SageMaker request: " + str(e))

    except sagemaker_runtime.exceptions.ModelError as e:
        logger.error(f"SageMaker model error: {e}")
        return error_response(500, "SageMaker model error: " + str(e))

    except Exception as e:
        logger.error(f"Internal server error: {e}")
        return error_response(500, "Internal server error: " + str(e))


def error_response(status_code, message):
    return {
        'statusCode': status_code,
        'body': json.dumps({'error': message})
    }
