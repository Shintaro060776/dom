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
                return {
                    'statusCode': 200,
                    'body': json.dumps({'video_url': video_url})
                }
            elif response.status_code == 202:
                return {
                    'statusCode': 202,
                    'body': json.dumps({'message': 'Video generation in progress'})
                }
            else:
                return {
                    'statusCode': response.status_code,
                    'body': json.dumps({'error': response.text})
                }

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }
    else:
        error_msg = "Invalid event format, 'generation_id' not found"
        logger.error(error_msg)
        return {
            'statusCode': 400,
            'body': json.dumps({'error': error_msg})
        }
