import json
import  boto3
import traceback
import uuid
from botocore.client import Config

def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    table_name = 'ImageMetadata'
    table = dynamodb.Table(table_name)

    s3 = boto3.client("s3", config=Config(signature_version='s3v4'))
    bucket_name = 'capture20090317'
    expiration = 300

    try:
        body_str = event.get('body', '{}')
        if body_str is None:
            body_str = '{}'
        print("Received body:", body_str)

        body = json.loads(body_str)

        if 'fileType' in body:
            unique_id = str(uuid.uuid4())
            file_type = body['fileType']
            file_name = f"image_{unique_id}.png"

            table.put_item(
                Item={
                    'id': unique_id,
                    'url': f"https://{bucket_name}.s3.amazonaws.com/{file_name}",
                    'timestamp': int(uuid.uuid4().time_low)
                }
            )

            response = s3.generate_presigned_url('put_object',
                                                Params={
                                                    'Bucket': bucket_name,
                                                    'Key': file_name,
                                                    'ACL': 'bucket-owner-full-control',
                                                    'ContentType': file_type},
                                                    ExpiresIn=expiration,
                                                    HttpMethod="PUT")

            return {
                'statusCode': 200,
                'body': json.dumps({'presignedUrl': response, 'fileName': file_name})
            }
        else:
            print("Missing required parameter in the request body")
            return {
                'statusCode': 400,
                'body': json.dumps('Missing or invalid parameters: "fileType" os required ')
            }
    except json.JSONDecodeError as e:
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