import React from 'react';
import './13.css';

const BlogArticle13 = () => {

    const pythonCode = `
    import json
    import boto3
    import requests
    import base64
    from botocore.exceptions import ClientError
    import os
    
    dynamodb = boto3.resource('dynamodb')
    translate = boto3.client('translate')
    s3 = boto3.client('s3')
    table_name = 'image2image20090317'
    bucket_name = 'image2image20090317'
    
    
    def lambda_handler(event, context):
        print("lambda function started")
        try:
            file_name = event['Records'][0]['s3']['object']['key']
        except Exception as e:
            print(f"Error getting file_name: {e}")
            return {'statusCode': 400, 'body': json.dumps('Error parsing event data: ' + str(e))}
    
        try:
            table = dynamodb.Table(table_name)
            response = table.get_item(Key={'FileName': file_name})
            if 'Item' not in response:
                return {'statusCode': 404, 'body': json.dumps('Prompt not found')}
        except ClientError as e:
            print(f"Error getting DB info: {e}")
            return {'statusCode': 500, 'body': json.dumps('Error accessing DynamoDB' + str(e))}
    
        item = response['Item']
        print(f"Retrieved item from DynamoDB: {item}")
        main_prompt = item['TranslatedPrompt']
        additional_prompt = item.get('AdditionalTranslatedPrompt', '')
        # init_image_path = f"s3://{bucket_name}/{file_name}"
    
        try:
            generated_image = call_stabilityai_api(
                main_prompt, additional_prompt, bucket_name, file_name)
        except Exception as e:
            print(f"Error invoking StabilityAI API: {e}")
            return {'statusCode': 500, 'body': json.dumps('Error calling stabilityAI API:' + str(e))}
    
        try:
            output_file_name = f"gen/generated_{file_name}"
            for generated_image in generated_image:
                s3.put_object(Bucket=bucket_name, Key=output_file_name,
                              Body=generated_image)
        except ClientError as e:
            print(f"Error putting image to S3: {e}")
            return {'statusCode': 500, 'body': json.dumps('Error saving image to S3: ' + str(e))}
    
        return {'statusCode': 200, 'body': json.dumps('Image generated Successful')}
    
    
    def translate_text(text, source_language, target_language):
        try:
            response = translate.translate_text(
                Text=text,
                SourceLanguageCode=source_language,
                TargetLanguageCode=target_language
            )
    
            return response['TranslatedText']
        except ClientError as e:
            raise Exception(f"Error translating text: {str(e)}")
    
    
    def call_stabilityai_api(main_prompt, additional_prompt, bucket_name, file_key):
        print(
            f"call_stabilityai_api called with: {main_prompt}, {additional_prompt}, {bucket_name}, {file_key}")
    
        translated_main_prompt = translate_text(main_prompt, 'ja', 'en')
        translated_additional_prompt = translate_text(
            additional_prompt, 'ja', 'en') if additional_prompt else ''
    
        api_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
        api_key = os.environ['STABILITY_API_KEY']
    
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
        try:
            response = s3.get_object(Bucket=bucket_name, Key=file_key)
            init_image_data = response['Body'].read()
        except ClientError as e:
            print(f"Fetching image: {e}")
            raise Exception('Error fetching image from S3: ' + str(e))
    
        request_data = {
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": 0.35,
            "steps": 40,
            "seed": 0,
            "cfg_scale": 5,
            "samples": 1,
            "text_prompts[0][text]": translated_main_prompt,
            "text_prompts[0][weight]": 1
        }
    
        if translated_additional_prompt:
            request_data["text_prompts[1][text]"] = translated_additional_prompt
            request_data["text_prompts[1][weight]"] = -1
    
        try:
            response = requests.post(api_url, headers=headers, files={
                                     "init_image": ("filename", init_image_data)}, data=request_data)
    
            if response.status_code != 200:
                raise Exception("Non-200 response:" + response.text)
        except Exception as e:
            raise Exception('Error calling StabilityAI API: ' + str(e))
    
        data = response.json()
        generated_images = [base64.b64decode(
            image['base64']) for image in data["artifacts"]]
    
        return generated_images
    
        `;

    return (
        <div className='App'>
            <img src='/blog/generated_20240127_06_47_0.png' alt='twelveth' className='header-image' />
            <div className='page-title'>
                <h1>Image-to-Image StabilityAI</h1>
            </div>
            <div className='page-date'>
                <p>2023/10/29</p>
            </div>
            <div className='paragraph'>
                <p>
                    Image-to-Image StabilityAI<br /><br />

                    今回は、StabilityAIの、Image2Imageのモデルを利用した、アプリケーションの、説明を、以下に記載します。<br /><br />

                    <img src='/blog/system12.png' alt='twelvethsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、アップロードする画像ファイルと、どのように、Stabilityaiに、画像を変換してほしいかを指示する、プロンプトを実装<br /><br />
                    プリサインURLを、Lambdaにて、発行してもらう為に、React == Nodejs == API Gateway == プリサインURL発行用の、Lambda、という通信経路で、Lambda関数を実行<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    フロントからの、APIリクエストを受けて、Apigatewayに転送します。<br /><br />
                    Stabilityaiで、生成された画像を、直接、取得するようにしています。<br /><br />


                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">S3</span><br /><br />
                    Lambdaにて、発行されたプリサインURLを、フロントに返して、そのプリサインURL向けに、フロントは、S3に画像をアップロードします<br /><br />
                    画像が、S3にアップロードされたら、それをトリガーにして、もう一つのLambda関数を、実行します。<br /><br />

                    <br /><br /><span className="highlight">プリサインURL発行用のLambda関数</span><br /><br />
                    Lambdaにて、プリサインURLを、発行します。<br /><br />
                    ユーザーからの、APIリクエストに、プロンプト情報も、このLambda関数に転送するようにしているので、そのプロンプトを、DynamoDBに書き込みます。<br /><br />

                    <br /><br /><span className="highlight">StabilityaiのAPI向けのLambda関数</span><br /><br />
                    S3のイベントトリガーにより、このLambda関数は、実行されます。<br /><br />
                    AWSのTranslateにより、DynamoDBに保存された、プロンプトを、日本語から、英語に翻訳します。<br /><br />
                    そのプロンプト情報と、S3にアップロードされた画像を、Stabilityaiに、送信します。<br /><br />
                    Stabilityaiによって、生成された画像を、S3にアップロードします。<br /><br />
                    フロントで、GetImageボタンを押下して、Stabilityaiによって、生成された画像を、取得して、フロントに表示します。<br /><br />


                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/image2image.mp4" type="video/mp4" />
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

export default BlogArticle13;
