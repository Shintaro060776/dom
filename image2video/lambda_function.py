import requests
import boto3
import os
import json
import logging
import traceback
import base64
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)


ssm = boto3.client('ssm')


def get_parameter(param_name):
    response = ssm.get_parameter(Name=param_name, WithDecryption=True)
    return response['Parameter']['Value']


STABILITY_API_KEY = os.environ['STABILITY_API_KEY']
S3_BUCKET_NAME = os.environ['S3_BUCKET_NAME']
SLACK_WEBHOOK_URL = os.environ['SLACK_WEBHOOK_URL']


def send_slack_message(message):
    payload = {"text": message}
    response = requests.post(SLACK_WEBHOOK_URL, data=json.dumps(
        payload), headers={'Content-Type': 'application/json'})
    if response.status_code != 200:
        raise Exception(f"Failed to send Slack message: {response.text}")


def start_video_generation(image_data):
    try:
        response = requests.post(
            "https://api.stability.ai/v2alpha/generation/image-to-video",
            headers={"authorization": f"Bearer {STABILITY_API_KEY}"},
            data={"seed": 0, "cfg_scale": 2.5, "motion_bucket_id": 40},
            files={"image": ("file", image_data, "image/png")}
        )
        response.raise_for_status()
        return response.json()["id"]
    except requests.exceptions.HTTPError as e:
        raise Exception(
            f"HTTP error: {e.response.status_code} {e.response.text}")
    except Exception as e:
        raise Exception(f"Error starting video generation: {str(e)}")


def check_generation_status(generation_id):
    try:
        response = requests.get(
            f"https://api.stability.ai/v2alpha/generation/image-to-video/result/{generation_id}",
            headers={'authorization': f"Bearer {STABILITY_API_KEY}"}
        )
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 202:
            return e.response
        else:
            raise Exception(
                f"HTTP error: {e.response.status_code} {e.response.text}")
    except Exception as e:
        raise Exception(f"Error checking generation status: {str(e)}")


def parse_multipart_body(body, is_base64_encoded, headers):
    try:
        if is_base64_encoded:
            body = base64.b64decode(body)

        if isinstance(body, bytes):
            body = body.decode('utf-8')

        content_type_header = headers.get('Content-Type', '')
        boundary = content_type_header.split("boundary=")[1]
        boundary = '--' + boundary

        parts = body.split(boundary + '\r\n')[1:-1]

        for part in parts:
            header_part, data_part = part.split('\r\n\r\n', 1)
            if 'name="image"' in header_part:
                return data_part.strip()

        logger.error(
            "Image data not found in event body. Parts: {}".format(parts))
        raise KeyError("Image data not found in event body")

    except Exception as e:
        logger.error(f"Error in parse_multipart_body: {str(e)}")
        logger.error(f"Headers: {headers}")
        logger.error(f"Is base64 encoded: {is_base64_encoded}")
        logger.error(f"Body: {body}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def lambda_handler(event, context):
    logger.info("Received event: %s", json.dumps(event))
    try:
        headers = event.get('headers', {})
        body = event['body']
        is_base64_encoded = event.get('isBase64Encoded', False)

        image_data = parse_multipart_body(body, is_base64_encoded, headers)

        logger.info("Processing started...")

        generation_id = start_video_generation(image_data)
        send_slack_message(f"Video generation started: ID {generation_id}")

        response = check_generation_status(generation_id)
        if response.status_code == 202:
            send_slack_message(
                f"Generation in-progress for ID {generation_id}")
            return {"message": "Generation in-progress"}
        elif response.status_code == 200:
            video_content = response.content
            s3 = boto3.client('s3')
            s3.put_object(Bucket=S3_BUCKET_NAME,
                          Key=f"{generation_id}.mp4", Body=video_content)
            video_url = f"s3://{S3_BUCKET_NAME}/{generation_id}.mp4"
            send_slack_message(f"Generation complete! Video URL: {video_url}")
            return {"message": "Generation complete!", "video_url": video_url}
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())
        error_message = str(e)
        send_slack_message(f"Error in video generation: {error_message}")
        return {"error": error_message}
