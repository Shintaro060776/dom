import json
import boto3
import os


def lambda_handler(event, context):
    s3_client = boto3.client('s3')

    bucket_name = 'speech20090317'
    file_name = event['queryStringParameters']['file_name']

    expiration = 300

    try:
        response = s3_client.generate_presigned_url('put_object',
                                                    Params={
                                                        'Bucket': bucket_name, 'Key': file_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error generating presigned URL: {str(e)}')
        }

    return {
        'statusCode': 200,
        'body': json.dumps({'url': response})
    }
