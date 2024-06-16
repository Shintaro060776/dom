import React from 'react';
import './21.css';

const BlogArticle21 = () => {

    const pythonCode = `
    import json
    import boto3
    from decimal import Decimal

    class DecimalEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Decimal):
                return float(obj)
            return super(DecimalEncoder, self).default(obj)

    def lambda_handler(event, context):
        dynamodb = boto3.resource('dynamodb')
        table_name = 'ImageMetadata'
        table = dynamodb.Table(table_name)

        try:
            data = table.scan()
            items = data.get('Items', [])

            return {
                'statusCode': 200,
                'body': json.dumps(items, cls=DecimalEncoder)
            }
        except Exception as e:
            print(f"Error: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps(f'Internal Server Error: {str(e)}')
            }
        `;

    return (
        <div className='App'>
            <img src='/blog/drawing.png' alt='thirteeeenth' className='header-image' />
            <div className='page-title'>
                <h1>Image Gallery</h1>
            </div>
            <div className='page-date'>
                <p>2024/6/16</p>
            </div>
            <div className='paragraph'>
                <p>
                    Image Gallery<br /><br />

                    今回は、CANVAS上に描く絵を保存/表示させる、アプリケーションについて、以下に記載します。<br /><br />

                    <img src='/blog/system21.png' alt='thireenthsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、CANVAS上に、絵を描きます。<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    フロントからの、(POST)APIリクエストを受けて、Apigatewayに転送します。<br /><br />

                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">Dynamodbに画像情報を登録する/今まで描いた絵の画像をFetchするLambda関数</span><br /><br />
                    ユーザーが描いた絵の情報を、DynamoDBに保存し、別のLambda関数で、それらの画像全てをFetchします。<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/capture.mp4" type="video/mp4" />
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

export default BlogArticle21;
