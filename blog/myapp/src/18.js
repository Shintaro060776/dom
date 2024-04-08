import React from 'react';
import './18.css';

const BlogArticle18 = () => {

    const pythonCode = `
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
                UpdateExpression='SET #title = :title, #event_body = :event_body',
                ExpressionAttributeNames={
                    '#title': 'title',
                    '#event_body': 'body',
                },
                ExpressionAttributeValues={
                    ':title': title,
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
    
        `;

    return (
        <div className='App'>
            <img src='/blog/event.jpg' alt='eighteenth' className='header-image' />
            <div className='page-title'>
                <h1>Scheduling Event</h1>
            </div>
            <div className='page-date'>
                <p>2024/4/8</p>
            </div>
            <div className='paragraph'>
                <p>
                    Scheduling Event<br /><br />

                    今回は、イベントを登録する、アプリケーションの説明を、以下に記載します。<br /><br />

                    <img src='/blog/system18.png' alt='eighteenthsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、イベントを登録できるUIを提供する<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    フロントからの、(POST)APIリクエストを受けて、Apigatewayに転送します。<br /><br />

                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">Dynamodbにイベント情報を登録するLambda関数</span><br /><br />
                    Dynamodbに登録したい、イベントを、Lambda関数から、登録します<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/event.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video><br /><br />

                    <br /><br />以下は、忘備録として、Pythonのコードの説明を記載します。<br /><br />

                    <div class="code-box">
                        <code>
                            <pre><code>{pythonCode}</code></pre>
                        </code >
                    </div >
                </p >
            </div >
        </div >
    );
};

export default BlogArticle18;
