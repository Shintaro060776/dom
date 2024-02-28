import json
import boto3
from boto3.dynamodb.conditions import Key

TABLE_NAME = 'speech'

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(TABLE_NAME)


def lambda_handler(event, context):
    file_key = event['queryStringParameters']['fileKey']

    response = table.get_item(
        Key={
            'file_key': file_key
        }
    )

    if 'Item' in response:
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(response['Item'])
        }
    else:
        return {
            'statusCode': 404,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'message': 'Item not found'})
        }
