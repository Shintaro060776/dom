import json
import boto3
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Event')

def lambda_handler(event, context):
    http_method = event.get('httpMethod', '')

    if http_method == 'POST':
        return create_event(event)
    elif http_method == 'GET':
        return get_event(event)
    elif http_method == 'PUT':
        return update_event(event)
    elif http_method == 'DELETE':
        return delete_event(event)
    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Unsupported HTTP method')
        }

def create_event(event):
    body = json.loads(event.get('body', '{}'))
    event_id = body.get('id')
    title = body.get('title')

    try:
        table.put_item(
            Item={
                'id': event_id,
                'title': title,
                'body': body,
            }
        )

        return {'statusCode': 200, 'body': json.dumps('Event created successfully')}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps(str(e))}

def get_event(event):
    event_id = event['queryStringParameters']['id']
    try:
        response = table.get_item(Key={'id': event_id})
        if 'Item' in response:
            return {'statusCode': 200, 'body': json.dumps(response['Item'])}
        else:
            return {'statusCode': 404, 'body': json.dumps('Event not found')}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps(str(e))}

def update_event(event):
    body = json.loads(event.get('body', '{}'))
    event_id = body.get('id')
    title = body.get('title')
    event_body = body.get('body')

    try:
        response = table.update_item(
            Key={'id': event_id},
            UpdateExpression='SET title = :title, #event_body = :event_body',
            ExpressionAttributeNames={
                '#event_body': 'body',
            },
            ExpressionAttributeValues={
                'title': title,
                ':event_body': event_body,
            },
            ReturnValues='UPDATED_NEW'
        )

        return {'statusCode': 200, 'body': json.dumps('Event updated successfully')}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps(str(e))}

def delete_event(event):
    event_id = json.loads(event.get('body', '{}')).get('id')
    try:
        table.delete_item(Key={'id': event_id})
        return {'statusCode': 200, 'body': json.dumps('Event deleted successfully')}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps(str(e))}
