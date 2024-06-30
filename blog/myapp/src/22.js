import React from 'react';
import './22.css';

const BlogArticle22 = () => {

    const pythonCode = `
import json
import os
import requests
import boto3
import time

STABILITY_API_KEY = os.environ['STABILITY_API_KEY']
S3_BUCKET_NAME = os.environ['S3_BUCKET_NAME']
SLACK_WEBHOOK_URL = os.environ['SLACK_WEBHOOK_URL']
S3_FOLDER_PREFIX = "generated_images/"  # プレフィックスの設定

s3 = boto3.client('s3')

def lambda_handler(event, context):
    body = json.loads(event['body'])
    prompt = body['prompt']

    image_data = generate_image(prompt)

    if image_data:
        image_url = save_image_to_s3(image_data)

        send_image_to_slack(image_url)

        return {
            'statusCode': 200,
            'body': json.dumps({'imageUrl': image_url})
        }
    else:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to generate image'})
        }

def generate_image(prompt):
    url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*"
    }

    files = {
        'prompt': (None, prompt),
        'output_format': (None, 'webp')
    }

    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return response.content
    else:
        print(f"Error generating image: {response.text}")
        return None

def save_image_to_s3(image_data):
    image_name = f"{S3_FOLDER_PREFIX}generated_{int(time.time())}.webp"  # プレフィックスを追加
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=image_name,
            Body=image_data,
            ContentType='image/webp'
        )

        image_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{image_name}"
        return image_url
    except Exception as e:
        print(f"Error saving image to S3: {e}")
        return None

def send_image_to_slack(image_url):
    message = {
        "text": "Generated Image",
        "attachments": [
            {
                "title": "Here is your generated image",
                "image_url": image_url
            }
        ]
    }

    response = requests.post(SLACK_WEBHOOK_URL, data=json.dumps(message), headers={'Content-Type': 'application/json'})

    if response.status_code != 200:
        print(f"Error sending image to Slack: {response.text}")
        `;

    return (
        <div className='App'>
            <img src='/blog/StabilityAILatest.png' alt='foureeeenth' className='header-image' />
            <div className='page-title'>
                <h1>Image Gen By StabilityAI Latest Model</h1>
            </div>
            <div className='page-date'>
                <p>2024/7/1</p>
            </div>
            <div className='paragraph'>
                <p>
                    Image Gen By StabilityAI Latest Model<br /><br />

                    今回は、最新のStabilityAIのモデルを利用して、プロンプトから、画像を生成させるアプリケーションについて、以下に記載します。<br /><br />

                    <img src='/blog/system22.png' alt='foureenthsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、生成させたい画像のイメージを、入力します<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    フロントからの、(POST)APIリクエストを受けて、Apigatewayに転送します。<br /><br />
                    フロントからの、(GET)APIリクエストを受けて、直接、S3から、StabilityAIにて生成された画像を、取得します<br /><br />

                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">StabilityAIのAPIを叩くLambda関数</span><br /><br />
                    StabilityAIのAPIを叩いて、画像を生成させる処理をします。<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/prompt_gen.mp4" type="video/mp4" />
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

export default BlogArticle22;
