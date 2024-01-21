import json
import boto3
import requests
import os


def lambda_handler(event, context):
    s3 = boto3.client('s3')
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    object_key = event['Records'][0]['s3']['object']['key']

    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response['Body'].read()

        api_url = 'https://api.stability.ai/v2alpha/generation/image-to-video'
        api_key = os.environ['STABILITY_API_KEY']

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'multipart/form-data'
        }
        data = {
            'seed': 0,
            'cfg_scale': 2.5,
            'motion_bucket_id': 40
        }

        file_extension = object_key.split('.')[-1]
        mime_type = 'image/jpeg' if file_extension.lower() == 'jpg' else 'image/png'
        files = {'image': (object_key, image_data, mime_type)}

        response = requests.post(
            api_url, headers=headers, data=data, files=files)
        response.raise_for_status()

        generation_id = response.json().get('id')
        if generation_id:
            return {
                'statusCode': 200,
                'body': json.dumps({'generation_id': generation_id})
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to start video generation'})
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
