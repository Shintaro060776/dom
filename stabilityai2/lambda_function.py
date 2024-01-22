import json
import boto3
import requests
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):

    logger.info(f"Received event: {event}")

    if 'generation_id' in event:
        generation_id = event['generation_id']
        api_url = f'https://api.stability.ai/v2alpha/generation/image-to-video/result/{generation_id}'
        api_key = os.environ['STABILITY_API_KEY']
        s3_bucket_for_video = 'image2video20090317'

        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json'
            }

            response = requests.get(api_url, headers=headers)

            if response.status_code == 200:
                s3 = boto3.client('s3')
                video_data = response.content
                video_key = f'videos/{generation_id}.mp4'
                s3.put_object(Bucket=s3_bucket_for_video,
                              Key=video_key, Body=video_data)
                video_url = f'https://{s3_bucket_for_video}.s3.amazonaws.com/{video_key}'
                return {'statusCode': 200, 'body': json.dumps({'video_url': video_url})}
            elif response.status_code == 202:
                return {'statusCode': 202, 'body': json.dumps({'message': 'Video generation in progress'})}
            else:
                return {'statusCode': response.status_code, 'body': json.dumps({'error': response.text})}
        except Exception as e:
            print(f"Error: {str(e)}")
            return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

    elif 'Records' in event:
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
            print(f"API Response Status Code: {api_response.status_code}")
            print(f"API Response Headers: {api_response.headers}")
            print(f"API Response Body: {api_response.text}")

            api_response.raise_for_status()

            generation_id = api_response.json().get('id')
            if generation_id:
                sfn_client = boto3.client('stepfunctions')
                state_machine_arn = 'arn:aws:states:ap-northeast-1:715573459931:stateMachine:VideoGenerationStateMachine'
                sfn_response = sfn_client.start_execution(
                    stateMachineArn=state_machine_arn, input=json.dumps({'generation_id': generation_id}))
                return {'statusCode': 200, 'body': json.dumps({'generation_id': generation_id, 'executionArn': sfn_response['executionArn']})}
            else:
                return {'statusCode': 500, 'body': json.dumps({'error': 'Failed to start video generation'})}
        except Exception as e:
            print(f"Error: {str(e)}")
            return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

    else:
        error_msg = "Invalid event format, 'generation_id' not found"
        logger.error(error_msg)
        return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid event format'})}
