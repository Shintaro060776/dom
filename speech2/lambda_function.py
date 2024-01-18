import json
import boto3
import traceback


def lambda_handler(event, context):
    s3_client = boto3.client('s3')
    bucket_name = 'speech20090317'
    expiration = 300

    try:
        if event.get('queryStringParameters') and 'fileName' in event['queryStringParameters']:
            file_name = event['queryStringParameters']['fileName']
            response = s3_client.generate_presigned_url('put_object',
                                                        Params={
                                                            'Bucket': bucket_name, 'Key': file_name},
                                                        ExpiresIn=expiration)
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
