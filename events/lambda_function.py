import json
import boto3
from boto3.dynamodb.conditions import Key
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Event')

def lambda_handler(event, context):
    http_method = event.get('httpMethod', '')
    logger.info(f"Received event: {event}")

    if http_method == 'POST':
        return create_event(event)
    elif http_method == 'GET':
        return get_event(event)
    elif http_method == 'PUT':
        return update_event(event)
    elif http_method == 'DELETE':
        return delete_event(event)
    else:
        logger.error(f"Unsupported HTTP method: {http_method}")
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
    queryStringParameters = event.get('queryStringParameters')
    if queryStringParameters is None or 'id' not in queryStringParameters:
        return get_all_events()

        logger.error("Missing id parameter in query string")
        return {
            'statusCode': 400,
            'body': json.dumps('Missing id parameter')
        }

    event_id = queryStringParameters['id']
    logger.info(f"Fetching event with ID: {event_id}")

    try:
        response = table.get_item(Key={'id': event_id})
        if 'Item' in response:
            logger.info(f"Found event: {response['Item']}")
            return {
                'statusCode': 200,
                'body': json.dumps(response['Item'])
            }
        else:
            logger.info(f"Event not found: {event_id}")
            return {
                'statusCode': 404,
                'body': json.dumps('Event not found')
            }
    except Exception as e:
        logger.error(f"Error fetching event: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(str(e))
        }

def get_all_events():
    try:
        response = table.scan()
        items = response.get('Items', [])
        return {
            'statusCode': 200,
            'body': json.dumps(items)
        }
    except Exception as e:
        logger.error(f"Error fetching all events: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(str(e))
        }

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
