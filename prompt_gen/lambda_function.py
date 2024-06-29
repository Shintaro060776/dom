import json
import os
import requests
import boto3

STABILITY_API_KEY = os.environ['STABILITY_API_KEY']
S3_BUCKET_NAME = os.environ['S3_BUCKET_NAME']
SLACK_WEBHOOK_URL = os.environ['SLACK_WEBHOOK_URL']

s3 = boto3.client('s3')

def lambda_handler(event, context):
    body = json.loads(event['body'])
    prompt = body['prompt']

    image_data = generate_image(prompt)

    if image_data:
        image_url = save_image_to_s3(image_data)

        send_image_to_slack(image_url)

        return {
            'statusCode': 200,
            'body': json.dumps({'imageUrl': image_url})
        }
    else:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to generate image'})
        }

def generate_image(prompt):
    url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*"
    }

    data = {
        "prompt": prompt,
        "output_format": "webp"
    }

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        return response.content
    else:
        print(f"Error generating image: {response.text}")
        return None

def save_image_to_s3(image_data):
    image_name = f"generated_{int(time.time())}.webp"
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=image_name,
            Body=image_data,
            ContentType='image/webp'
        )

        image_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{image_data}"
        return image_url
    except Exception as e:
        print(f"Error saving image to S3: {e}")
        return None

def send_image_to_slack(image_url):
    message = {
        "text": "Generated Image",
        "attachments": [
            {
                "title": "Here is your generated image",
                "image_url": image_url
            }
        ]
    }

    response = requests.post(SLACK_WEBHOOK_URL, data=json.dumps(message), headers={'Content-Type': 'application/json'})

    if response.status_code != 200:
        print(f"Error sending image to slack: {response.text}")