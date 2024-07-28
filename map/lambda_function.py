import json
import boto3
from datetime import datetime
import uuid

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Routes')

def lambda_handler(event, context):
    try:
        print('Event:', event)
        body = json.loads(event.get('body', '{}'))
        user_id = body.get('userId')
        route_data = body.get('routeData')

        if not user_id or not route_data:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Invalid input data'})
            }

        route_id = str(uuid.uuid4())

        timestamp = datetime.now().isoformat()

        table.put_item(
            Item={
                'userId': user_id,
                'routeId': route_id,
                'routeData': route_data,
                'timestamp': timestamp
            }
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Route saved successfully',
                'routeId': route_id
            })
        }

    except Exception as e:
        print('Error:', e)
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Error saving route'})
        }