import boto3
import requests
import os
from botocore.exceptions import ClientError

STABILITY_AI_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/edit/search-and-replace"
STABILITY_AI_API_KEY = os.environ['STABILITY_API_KEY']

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

DYNAMODB_TABLE_NAME = 'searchandreplace20090317'
OUTPUT_BUCKET_NAME = 'searchandreplace20090317'

def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']

    try:
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        response = table.get_item(Key={'FileName': file_key})
        if 'Item' not in response:
            return {'statusCode': 404, 'body': 'Item not found in DynamoDB'}

        prompt = response['Item']['Prompt']
        search_prompt = response['Item']['SearchPrompt']

        image_response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        image_content = image_response['Body'].read()

        response = requests.post(
            STABILITY_AI_ENDPOINT,
            headers={
                "Authorization": STABILITY_AI_API_KEY,
            },
            files={
                'image': ('filename', image_content, 'image/png')
            },
            data={
                'prompt': prompt,
                'search_prompt': search_prompt,
                'output_format': 'png',
            },
        )

        if response.status_code == 200:
            output_file_key = f"gen/{file_key}" 
            s3_client.put_object(Bucket=OUTPUT_BUCKET_NAME, Key=output_file_key, Body=response.content, ContentType='image/png')

            return {'statusCode': 200, 'body': 'Image generated and saved successfully'}
        else:
            return {'statusCode': response.status_code, 'body': 'Failed to generate image'}

    except ClientError as e:
        print(e)
        return {'statusCode': 500, 'body': 'Internal Server Error'}