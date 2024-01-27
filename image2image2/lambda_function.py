import json
import boto3
import requests
import base64
from botocore.exceptions import ClientError

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
table_name = 'image2image20090317'
bucket_name = 'image2image20090317'


def lambda_handler(event, context):
    file_name = event['Records'][0]['s3']['object']['key']

    try:
        table = dynamodb.Table(table_name)
        response = table.get_item(Key={'FileName': file_name})
        if 'Item' not in response:
            return {'statusCode': 404, 'body': json.dumps('Prompt not found')}

        item = response['Item']
        translated_prompt = item['TranslatedPrompt']

        generated_image = call_stabilityai_api(translated_prompt)

        output_file_name = f"gen/generated_{file_name}"
        s3.put_object(Bucket=bucket_name, Key=output_file_name,
                      Body=generated_image)

        return {'statusCode': 200, 'body': json.dumps('Image generated Successful')}
    except ClientError as e:
        print(e)
        return {'statusCode': 500, 'body': json.dumps('Internal Server Error')}


def call_stabilityai_api(main_prompt, additional_prompt, init_image_path):
    api_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
    api_key = ["STABILITYAI_API_KEY"]

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    with open(init_image_path, "rb") as init_image_file:
        response = requests.post(
            api_url,
            headers=headers,
            files={"init_image": init_image_file},
            data={
                "init_image_mode": "IMAGE_STRENGTH",
                "image_strength": 0.35,
                "steps": 40,
                "width": 1024,
                "height": 1024,
                "seed": 0,
                "cfg_scale": 5,
                "samples": 1,
                "text_prompts[0][text]": main_prompt,
                "text_prompts[0][weight]": 1,
                "text_prompts[1][text]": additional_prompt,
                "text_prompts[1][weight]": -1,
            }
        )

        if response.status_code != 200:
            raise Exception("Non-200 response:" + response.text)

        data = response.json()
        generated_images = []

        for image in data["artifacts"]:
            generated_image = base64.b64decode(image["base64"])
            generated_images.append(generated_image)

        return generated_images
