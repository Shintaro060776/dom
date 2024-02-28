import json
import boto3
import requests
import base64
from botocore.exceptions import ClientError
import os

dynamodb = boto3.resource('dynamodb')
translate = boto3.client('translate')
s3 = boto3.client('s3')
table_name = 'image2image20090317'
bucket_name = 'image2image20090317'


def lambda_handler(event, context):
    print("lambda function started")
    try:
        file_name = event['Records'][0]['s3']['object']['key']
    except Exception as e:
        print(f"Error getting file_name: {e}")
        return {'statusCode': 400, 'body': json.dumps('Error parsing event data: ' + str(e))}

    try:
        table = dynamodb.Table(table_name)
        response = table.get_item(Key={'FileName': file_name})
        if 'Item' not in response:
            return {'statusCode': 404, 'body': json.dumps('Prompt not found')}
    except ClientError as e:
        print(f"Error getting DB info: {e}")
        return {'statusCode': 500, 'body': json.dumps('Error accessing DynamoDB' + str(e))}

    item = response['Item']
    print(f"Retrieved item from DynamoDB: {item}")
    main_prompt = item['TranslatedPrompt']
    additional_prompt = item.get('AdditionalTranslatedPrompt', '')
    # init_image_path = f"s3://{bucket_name}/{file_name}"

    try:
        generated_image = call_stabilityai_api(
            main_prompt, additional_prompt, bucket_name, file_name)
    except Exception as e:
        print(f"Error invoking StabilityAI API: {e}")
        return {'statusCode': 500, 'body': json.dumps('Error calling stabilityAI API:' + str(e))}

    try:
        output_file_name = f"gen/generated_{file_name}"
        for generated_image in generated_image:
            s3.put_object(Bucket=bucket_name, Key=output_file_name,
                          Body=generated_image)
    except ClientError as e:
        print(f"Error putting image to S3: {e}")
        return {'statusCode': 500, 'body': json.dumps('Error saving image to S3: ' + str(e))}

    return {'statusCode': 200, 'body': json.dumps('Image generated Successful')}


def translate_text(text, source_language, target_language):
    try:
        response = translate.translate_text(
            Text=text,
            SourceLanguageCode=source_language,
            TargetLanguageCode=target_language
        )

        return response['TranslatedText']
    except ClientError as e:
        raise Exception(f"Error translating text: {str(e)}")


def call_stabilityai_api(main_prompt, additional_prompt, bucket_name, file_key):
    print(
        f"call_stabilityai_api called with: {main_prompt}, {additional_prompt}, {bucket_name}, {file_key}")

    translated_main_prompt = translate_text(main_prompt, 'ja', 'en')
    translated_additional_prompt = translate_text(
        additional_prompt, 'ja', 'en') if additional_prompt else ''

    api_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
    api_key = os.environ['STABILITY_API_KEY']

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        init_image_data = response['Body'].read()
    except ClientError as e:
        print(f"Fetching image: {e}")
        raise Exception('Error fetching image from S3: ' + str(e))

    request_data = {
        "init_image_mode": "IMAGE_STRENGTH",
        "image_strength": 0.35,
        "steps": 40,
        "seed": 0,
        "cfg_scale": 5,
        "samples": 1,
        "text_prompts[0][text]": translated_main_prompt,
        "text_prompts[0][weight]": 1
    }

    if translated_additional_prompt:
        request_data["text_prompts[1][text]"] = translated_additional_prompt
        request_data["text_prompts[1][weight]"] = -1

    try:
        response = requests.post(api_url, headers=headers, files={
                                 "init_image": ("filename", init_image_data)}, data=request_data)

        if response.status_code != 200:
            raise Exception("Non-200 response:" + response.text)
    except Exception as e:
        raise Exception('Error calling StabilityAI API: ' + str(e))

    data = response.json()
    generated_images = [base64.b64decode(
        image['base64']) for image in data["artifacts"]]

    return generated_images
