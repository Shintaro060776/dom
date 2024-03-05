import json
import boto3

def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    table_name = 'MusicInformation'
    table = dynamodb.Table(table_name)

    try:
        response = table.scan()

        items = response['Item']

        return {
            'statusCode': 200,
            'body': json.dumps(items)
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps("Internal Server Error")
        }