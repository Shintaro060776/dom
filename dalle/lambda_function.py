import os
import json
from openai import OpenAI


def lambda_handler(event, context):
    api_key = os.environ['OPENAI_API_KEY']

    client = OpenAI(api_key=api_key)

    body = json.loads(event['body'])
    prompt = body['prompt']

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url

        return {
            'statusCode': 200,
            'body': json.dumps({'imageUrl': image_url})
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
