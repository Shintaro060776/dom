import React from 'react';
import './23.css';

const BlogArticle23 = () => {

    const pythonCode = `
import json
import boto3
from datetime import datetime
import uuid
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Routes')

def decimal_default(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    raise TypeError

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

        route_data = [[Decimal(str(lat)), Decimal(str(lng))] for lat, lng in route_data]

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
        `;

    return (
        <div className='App'>
            <img src='/blog/marathon.png' alt='marathon' className='header-image' />
            <div className='page-title'>
                <h1>Marathon Tracker</h1>
            </div>
            <div className='page-date'>
                <p>2024/8/7</p>
            </div>
            <div className='paragraph'>
                <p>
                    Marathon Tracker<br /><br />

                    今回は、走行距離を計測するアプリケーションについて、以下に記載します。<br /><br />

                    <img src='/blog/system23.png' alt='marathonsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、走ったルートを、(ダブルクリックして)フロント上に記録します<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    フロントからの、(POST)APIリクエストを受けて、Apigatewayに転送します。<br /><br />
                    フロントからの、(GET)APIリクエストを受けて、直接、DynamoDBから、取得します<br /><br />

                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">Lambda関数</span><br /><br />
                    走ったルートを、DynamoDBに保存するようにします。<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/map.mp4" type="video/mp4" />
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

export default BlogArticle23;
