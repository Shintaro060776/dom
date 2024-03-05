import json
import boto3
import traceback
import uuid
from botocore.client import Config

def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    table_name = 'MusicInformation'
    table = dynamodb.Table(table_name)

    s3 = boto3.client("s3", config=Config(signature_version='s3v4'))
    bucket_name = 'music20090317'
    expiration = 300

    try:
        body_str = event.get('body', '{}')
        if body_str is None:
            body_str = '{}'
        print("Received body:", body_str)

        body = json.loads(body_str)

        if 'title' in body and 'rating' in body and 'fileName' in body:
            unique_id = str(uuid.uuid4())
            title = body['title']
            rating = body['rating']
            file_name = unique_id + '_' + body['fileName']

            table.put_item(
                Item={
                    'id': unique_id,
                    'title': title,
                    'rating': rating,
                    'fileName': file_name,
                }
            )

            response = s3.generate_presigned_url('put_object',
                                                Params={
                                                    'Bucket': bucket_name,
                                                    'Key': file_name,
                                                    'ACL': 'bucket-owner-full-control'},
                                                    ExpiresIn=expiration,
                                                    HttpMethod="PUT")

            return {
                'statusCode': 200,
                'body': json.dumps({'url': response, 'id': unique_id})
            }
        else:
            print("Missing required parameters in the request body")
            return {
                'statusCode': 400,
                'body': json.dumps('Missing or invalid parameters: "title", "rating", and "fileName" are required')
            }
    except json.JSONDecodeErrror as e:
        print(f"JSON Decode Error: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid JSON format')
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Internal Server Error: {str(e)}')
        }
        