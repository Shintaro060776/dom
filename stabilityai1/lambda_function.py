import json
import boto3
import traceback
from botocore.client import Config


def lambda_handler(event, context):
    s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
    bucket_name = 'image2video20090317'
    expiration = 300

    try:
        body = json.loads(event.get('body', '{}'))

        if 'fileName' in body:
            file_name = body['fileName']
            response = s3.generate_presigned_url('put_object',
                                                 Params={
                                                     'Bucket': bucket_name,
                                                     'Key': file_name,
                                                     'ACL': 'bucket-owner-full-control'},
                                                 ExpiresIn=expiration,
                                                 HttpMethod="PUT")
            return {
                'statusCode': 200,
                'body': json.dumps({'url': response})
            }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps('Missing or invalid query parameters')
            }
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps(f'Internal server error: {str(e)}')
        }
