import json
import boto3

def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    table_name = 'ImageMetadata' 
    table = dynamodb.Table(table_name)

    try:
        params = {
            'TableName': table_name
        }

        data = table.scan(params)
        items = data.get('Items', [])

        return {
            'statusCode': 200,
            'body': json.dumps(items)
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Internal Server Error: {str(e)}')
        }