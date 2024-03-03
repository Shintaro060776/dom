import React from 'react';
import './15.css';

const BlogArticle15 = () => {

    const pythonCode = `
    import base64
    import os
    import requests
    import boto3
    from datetime import datetime
    import json
    from io import BytesIO
    
    # 環境変数から設定を読み込む
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    engine_id = "stable-diffusion-v1-6"
    api_key = os.getenv("STABILITY_API_KEY")
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    s3_bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if not api_key:
        raise Exception("Missing Stability API key.")
    
    if not slack_webhook_url:
        raise Exception("Missing Slack webhook URL.")
    
    # S3クライアントの初期化
    s3 = boto3.client('s3')
    
    def generate_image(text_prompt):
        try:
            response = requests.post(
                f"{api_host}/v1/generation/{engine_id}/text-to-image",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "text_prompts": [
                        {
                            "text": text_prompt,
                            "weight": 1
                        },
                        {
                            "text": "blurry, bad",
                            "weight": -1
                        },
                    ],
                    "cfg_scale": 5,
                    "height": 512,
                    "width": 512,
                    "samples": 1,
                    "steps": 40,
                    "seed": 0,
                },
            )
    
            response.raise_for_status()
            data = response.json()
            return data["artifacts"]
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise
    
    
    def lambda_handler(event, context):
        try:
            body = json.loads(event.get('body', '{}'))
    
            text_prompt = body.get('text')
    
            print("Received text_prompt:", text_prompt)
    
            images = generate_image(text_prompt)
            image_urls = []
    
            for i, image in enumerate(images):
                image_data = base64.b64decode(image["base64"])
                file_name = f"v1_txt2img_{datetime.now().isoformat()}_{i}.png"
                s3_path = f"generated-images/{file_name}"
    
                # S3に画像を保存
                image_file = BytesIO(image_data)
    
                s3.upload_fileobj(
                    Fileobj=image_file,
                    Bucket=s3_bucket_name,
                    Key=s3_path
                )
    
                # 生成された画像のS3 URL
                image_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_path}"
                image_urls.append(image_url)
    
                # Slackに通知
                requests.post(
                    slack_webhook_url,
                    json={"text": f"Generated image: {image_url}"}
                )
    
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "message": "Image generated and posted to Slack successdully",
                    "imageUrls": image_urls
                })
            }
        except Exception as e:
            print(f"Error occurred: {e}")
            return {
                "statusCode": 500,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({"message": "An error occurred during the image generation process"})
            }
        `;

    return (
        <div className='App'>
            <img src='/blog/v1_txt2img_2024-03-02T08_23_40.823434_0.png' alt='fifteenth' className='header-image' />
            <div className='page-title'>
                <h1>Text-to-Image StabilityAI</h1>
            </div>
            <div className='page-date'>
                <p>2024/1/28</p>
            </div>
            <div className='paragraph'>
                <p>
                    Text-to-Image StabilityAI<br /><br />

                    今回は、StabilityAIの、Text-to-Imageのモデルを利用した、アプリケーションの、説明を、以下に記載します。<br /><br />

                    <img src='/blog/system15.png' alt='fifteenthsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、StabilityAIに、生成してほしい画像の為の、プロンプトを、フロントで入力して、それをバックエンド側で、転送します。<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    フロントからの、(POST)APIリクエストを受けて、Apigatewayに転送します。<br /><br />

                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">StabilityAIのAPI向けのLambda関数</span><br /><br />
                    StabilityAIに生成してほしい画像のイメージを、StabilityAIに転送する<br /><br />
                    StabilityAIにて、生成する画像を、S3に保存するのと、Slackに画像を転送する<br /><br />


                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/shark.mp4" type="video/mp4" />
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

export default BlogArticle15;
