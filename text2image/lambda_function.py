import base64
import os
import requests
import boto3
from datetime import datetime

# 環境変数から設定を読み込む
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
engine_id = "stable-diffusion-v1-6"
api_key = os.getenv("STABILITY_API_KEY")
slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")

if not api_key:
    raise Exception("Missing Stability API key.")

if not slack_webhook_url:
    raise Exception("Missing Slack webhook URL.")

# S3クライアントの初期化
s3 = boto3.client('s3')

def generate_image(text_prompt):
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [{"text": text_prompt}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    return data["artifacts"]

def lambda_handler(event, context):
    # イベントからtext_promptを取得し、デフォルト値は設定しない
    if 'text_prompt' not in event:
        return {
            "statusCode": 400,
            "body": "Missing 'text_prompt' in request."
        }
    
    text_prompt = event['text_prompt']
    images = generate_image(text_prompt)

    for i, image in enumerate(images):
        image_data = base64.b64decode(image["base64"])
        file_name = f"v1_txt2img_{datetime.now().isoformat()}_{i}.png"
        s3_path = f"generated-images/{file_name}"

        # S3に画像を保存
        s3.upload_fileobj(
            Fileobj=bytes(image_data),
            Bucket=s3_bucket_name,
            Key=s3_path
        )

        # 生成された画像のS3 URL
        image_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_path}"

        # Slackに通知
        requests.post(
            slack_webhook_url,
            json={"text": f"Generated image: {image_url}"}
        )

    return {
        "statusCode": 200,
        "body": "Image generated and posted to Slack successfully."
    }