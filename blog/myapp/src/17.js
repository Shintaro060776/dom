import React from 'react';
import './17.css';

const BlogArticle17 = () => {

    const pythonCode = `
    import json
    import boto3
    
    def lambda_handler(event, context):
        dynamodb = boto3.resource('dynamodb')
        table_name = 'MusicInformation'
        table = dynamodb.Table(table_name)
    
        try:
            response = table.scan()
    
            items = response['Items']
    
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
        `;

    return (
        <div className='App'>
            <img src='/blog/music.jpg' alt='seventeenth' className='header-image' />
            <div className='page-title'>
                <h1>Music Rating</h1>
            </div>
            <div className='page-date'>
                <p>2024/3/8</p>
            </div>
            <div className='paragraph'>
                <p>
                    Music Rating<br /><br />

                    今回は、お気に入りの音楽について、評価するアプリケーションの説明を、以下に記載します。<br /><br />

                    <img src='/blog/system17.png' alt='seventeenthsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、お気に入りの音楽の、タイトル/評価/画像を選択して、プリサインURL経由で、S3にアップロードします<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    プリサインURL発行の為に、フロントからの、(POST)APIリクエストを受けて、Apigatewayに転送します。<br /><br />

                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">Dynamodbに音楽情報を登録するLambda関数と、プリサインURL発行の為のLambda関数</span><br /><br />
                    Dynamodbに登録したい、音楽情報(タイトル、評価、画像ファイル名)を、Lambda関数から、登録します<br /><br />
                    フロントから、直接、S3にアニメーションに変換させたい画像をアップロードする為の、プリサインURLを発行します<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/music.mp4" type="video/mp4" />
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

export default BlogArticle17;
