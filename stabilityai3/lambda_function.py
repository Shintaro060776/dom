import json
import boto3
import requests
import os
import time
import logging
import base64

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    logger.info(f"Received event: {event}")

    if isinstance(event, str):
        try:
            event = json.loads(event)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid event format'})}

    if 'generation_id' not in event:
        logger.error("Generation ID not found in the event")
        return {'statusCode': 400, 'body': json.dumps({'error': 'Generation ID not found'})}

    generation_id = event['generation_id']
    api_url = f'https://api.stability.ai/v2alpha/generation/image-to-video/result/{generation_id}'
    api_key = os.environ['STABILITY_API_KEY']
    s3_bucket_for_video = 'image2video20090317'

    while True:
        try:
            headers = {'Authorization': f'Bearer {api_key}',
                       'Accept': 'application/json'}
            response = requests.get(api_url, headers=headers)
            logger.info(f"API Response Status Code: {response.status_code}")
            logger.info(f"API Response Body: {response.text}")

            if response.status_code == 200:
                s3 = boto3.client('s3')
                video_key = f'videos/{generation_id}.mp4'

                video_json = response.json()
                video_base64 = video_json['video']

                video_data = base64.b64decode(video_base64)

                s3.put_object(Bucket=s3_bucket_for_video,
                              Key=video_key, Body=video_data)
                video_url = f'https://{s3_bucket_for_video}.s3.amazonaws.com/{video_key}'
                return {'statusCode': 200, 'body': json.dumps({'video_url': video_url})}

            elif response.status_code == 202:
                time.sleep(10)
                continue

            else:
                error_msg = f"API request failed: Status Code {response.status_code}, Body: {response.text}"
                logger.error(error_msg)
                return {'statusCode': response.status_code, 'body': json.dumps({'error': error_msg})}

        except requests.RequestException as e:
            logger.error(f"HTTP request error: {e}")
            return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

    return {'statusCode': 400, 'body': json.dumps({'error': 'Video generation status check failed'})}
