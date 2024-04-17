import React from 'react';
import './19.css';

const BlogArticle19 = () => {

    const pythonCode = `
    import boto3
    import requests
    import os
    from botocore.exceptions import ClientError
    import logging
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    STABILITY_AI_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/edit/search-and-replace"
    STABILITY_AI_API_KEY = os.environ['STABILITY_API_KEY']
    
    s3_client = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    
    DYNAMODB_TABLE_NAME = 'searchandreplace20090317'
    OUTPUT_BUCKET_NAME = 'searchandreplace20090317'
    
    def lambda_handler(event, context):
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
    
        try:
            table = dynamodb.Table(DYNAMODB_TABLE_NAME)
            response = table.get_item(Key={'FileName': file_key})
            if 'Item' not in response:
                logger.info("Item not found in DynamoDB")
                return {'statusCode': 404, 'body': 'Item not found in DynamoDB'}
    
            prompt = response['Item']['Prompt']
            search_prompt = response['Item']['SearchPrompt']
    
            image_response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            image_content = image_response['Body'].read()
    
            logger.info(f"Sending API Request to Stability AI")
    
            response = requests.post(
                STABILITY_AI_ENDPOINT,
                headers={
                    "Authorization": STABILITY_AI_API_KEY,
                    "Accept": "image/*"
                },
                files={
                    'image': ('filename', image_content, 'image/png')
                },
                data={
                    'prompt': prompt,
                    'search_prompt': search_prompt,
                    'output_format': 'png',
                },
            )
    
            logger.info(f"API Response Status Code: {response.status_code}")
    
            if response.status_code == 200:
                output_file_key = f"gen/{file_key}" 
                s3_client.put_object(Bucket=OUTPUT_BUCKET_NAME, Key=output_file_key, Body=response.content, ContentType='image/png')
    
                logger.info(f"Image saved successfully in {output_file_key}")
                return {'statusCode': 200, 'body': 'Image generated and saved successfully'}
            else:
                logger.error(f"Failed to generate image: {response.text}")
                return {'statusCode': response.status_code, 'body': 'Failed to generate image'}
    
        except ClientError as e:
            print(e)
            logger.error(f"Client Error: {e}")
            return {'statusCode': 500, 'body': 'Internal Server Error'}    
        `;

    return (
        <div className='App'>
            <img src='/blog/searchandreplace.png' alt='nighnteenth' className='header-image' />
            <div className='page-title'>
                <h1>Search And Replace</h1>
            </div>
            <div className='page-date'>
                <p>2024/4/17</p>
            </div>
            <div className='paragraph'>
                <p>
                    Search And Replace<br /><br />

                    今回は、StabilityAIが提供する、特定のセグメント(画像)を、別のセグメント(画像)に変換する、アプリケーションの説明を、以下に記載します。<br /><br />

                    <img src='/blog/system19.png' alt='nighnteenthsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、変換したい画像の為の、プロンプトを入力する<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    フロントからの、(POST/Get)APIリクエストを受けて、Apigatewayに転送します。<br /><br />

                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">Dynamodbにプロンプト情報を登録する/プリサインURLを発行するLambda関数</span><br /><br />
                    Dynamodbに登録したい、イベントを、Lambda関数から、登録します<br /><br />

                    <br /><br /><span className="highlight">StabilityAIにリクエストするLambda関数</span><br /><br />
                    画像と、プロンプトを、StabilityAIに送信する<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/searchandreplace.mp4" type="video/mp4" />
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

export default BlogArticle19;
