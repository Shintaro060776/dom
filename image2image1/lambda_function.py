import json
import boto3
from botocore.exceptions import ClientError
import datetime
from botocore.client import Config


def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    translate = boto3.client('translate')
    s3 = boto3.client("s3", config=Config(signature_version="s3v4"))

    bucket_name = 'image2image20090317'
    expiration = 300
    table_name = 'image2image20090317'

    try:
        body = json.loads(event['body'])
        japanese_prompt = body['prompt']
        additional_japanese_prompt = body.get('additionalPrompt', '')
        file_name = body['fileName']

        response = translate.translate_text(
            Text=japanese_prompt,
            SourceLanguageCode='ja',
            TargetLanguageCode='en'
        )

        translated_text = response['TranslatedText']

        translated_additional_text = ''
        if additional_japanese_prompt:
            additional_response = translate.translate_text(
                Text=additional_japanese_prompt,
                SourceLanguageCode='ja',
                TargetLanguageCode='en'
            )

            translated_additional_text = additional_response['TranslatedText']

        table = dynamodb.Table(table_name)
        table.put_item(
            Item={
                'FileName': file_name,
                'JapanesePrompt': japanese_prompt,
                'TranslatedPrompt': translated_text,
                'AdditionalJapanesePrompt': additional_japanese_prompt,
                'TranslatedPrompt': translated_additional_text,
                'Timestamp': datetime.datetime.now().isoformat()
            }
        )

        presigned_url = s3.generate_presigned_url(
            ClientMethod='put_object',
            Params={
                'Bucket': bucket_name,
                'Key': file_name,
                'ACL': 'bucket-owner-full-control'
            },
            ExpiresIn=expiration,
            HttpMethod="PUT"
        )

        return {
            'statusCode': 200,
            'body': json.dumps({'url': presigned_url})
        }
    except ClientError as e:
        print(e)
        return {
            'statusCode': 500,
            'body': json.dumps('Internal server error')
        }
