import json
import boto3
from botocore.exceptions import ClientError
import datetime
from botocore.client import Config

def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    s3 = boto3.client("s3", config=Config(signature_version='s3v4'))

    bucket_name = 'searchandreplace20090317'
    expiration = 300
    table_name = 'searchandreplace20090317'

    try:
        body = json.loads(event['body'])
        prompt = body['prompt']
        search_prompt = body['search_prompt']
        file_name = body['fileName']

        table = dynamodb.Table(table_name)
        table.put_item(
            Item={
                'FileName': file_name,
                'Prompt': prompt,
                'SearchPrompt': search_prompt,
                'Timestamp': datetime.datetime.now().isoformat()
            }
        )

        presigned_url = s3.generate_presigned_url(
            ClientMethod = 'put_object',
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
            'body': json.dumps('Internal Server Error')
        }
        