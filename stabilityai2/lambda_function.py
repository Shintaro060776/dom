import json
import boto3
import requests
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    logger.info(f"Received event: {event}")

    if 'Records' in event:
        s3 = boto3.client('s3')
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        object_key = event['Records'][0]['s3']['object']['key']

        try:
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            image_data = response['Body'].read()

            api_url = 'https://api.stability.ai/v2alpha/generation/image-to-video'
            api_key = os.environ['STABILITY_API_KEY']

            headers = {'Authorization': f'Bearer {api_key}'}
            data = {'seed': 0, 'cfg_scale': 2.5, 'motion_bucket_id': 40}
            file_extension = object_key.split('.')[-1]
            mime_type = 'image/jpeg' if file_extension.lower() == 'jpg' else 'image/png'
            files = {'image': (object_key, image_data, mime_type)}

            api_response = requests.post(
                api_url, headers=headers, data=data, files=files)
            logger.info(
                f"API Response Status Code: {api_response.status_code}")
            logger.info(f"API Response Headers: {api_response.headers}")
            logger.info(f"API Response Body: {api_response.text}")

            api_response.raise_for_status()

            generation_id = api_response.json().get('id')
            if generation_id:
                sfn_client = boto3.client('stepfunctions')
                state_machine_arn = 'arn:aws:states:ap-northeast-1:715573459931:stateMachine:VideoGenerationStateMachine'
                sfn_response = sfn_client.start_execution(
                    stateMachineArn=state_machine_arn,
                    input=json.dumps({'generation_id': generation_id})
                )
                return {
                    'statusCode': 200,
                    'body': json.dumps({'generation_id': generation_id, 'executionArn': sfn_response['executionArn']})
                }
            else:
                return {
                    'statusCode': 500,
                    'body': json.dumps({'error': 'Failed to start video generation'})
                }

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }

    elif 'generation_id' in event:
        return {
            'statusCode': 200,
            'body': json.dumps(event)
        }

    else:
        return {'statusCode': 200, 'body': json.dumps({'message': 'Triggered by Step Functions, no action required'})}
