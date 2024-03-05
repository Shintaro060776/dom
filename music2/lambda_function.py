import json
import boto3
import os

def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    table_name = 'MusicInformation'
    table = dynamodb.Table(table_name)

    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']

        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"

        unique_id = object_key.split('_')[0]

        try:
            response = table.get_item(Key={'id': unique_id})

            if 'Item' in response:
                item = response['Item']

                item['imageUrl'] = s3_url

                table.put_item(Item=item)

                print(f"Updated DynamoDB record with ID {unique_id} to include image URL: {s3_url}")
            else:
                print(f"No matching record found in DynamoDB for ID {unique_id}")

        except Exception as e:
            print(f"Error updating DynamoDB record: {str(e)}")
            raise e

    return {
        'statusCode': 200,
        'body': json.dumps('Successfully processed S3 event')
    }