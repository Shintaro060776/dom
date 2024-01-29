import React from 'react';
import './14.css';

const BlogArticle14 = () => {

    const pythonCode = `
    import json
    from openai import OpenAI
    import os
    import boto3
    import logging
    
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    
    def lambda_handler(event, context):
    
        s3_client = boto3.client('s3')
    
        bucket_name = "text2speech20090317"
        s3_file_path = "speech.mp3"
    
        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
            user_input = event.get("user_input", "")
    
            try:
                GPT_MODEL = "gpt-4"
                messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": user_input},
                ]
                response = client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=messages,
                    temperature=0
                )
    
                generated_text = response.choices[0].message.content
                logger.info(f"AI Response: {generated_text}")
    
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                raise e
    
            speech_file_path = "/tmp/speech.mp3"
    
            try:
                speech_response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=generated_text
                )
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                raise e
    
            speech_response.stream_to_file(speech_file_path)
    
            try:
                with open(speech_file_path, 'rb') as speech_file:
                    s3_client.upload_fileobj(
                        speech_file, bucket_name, s3_file_path)
            except boto3.exceptions.S3UploadFailedError as e:
                print("Error uploading file to S3:", e)
                return {"statusCode": 500, "body": json.dumps({"error": "S3 upload error:" + str(e)})}
    
            audio_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_file_path}"
    
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": generated_text,
                    "audio_url": audio_url
                })
            }
    
        except Exception as e:
            print(e)
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)})
            }
    
    
        `;

    return (
        <div className='App'>
            <img src='/blog/20240128_10_30_0.png' alt='thiteenth' className='header-image' />
            <div className='page-title'>
                <h1>Text-to-Speech OpenAI</h1>
            </div>
            <div className='page-date'>
                <p>2023/11/28</p>
            </div>
            <div className='paragraph'>
                <p>
                    Text-to-Speech OpenAI<br /><br />

                    今回は、OpenAIの、Text-to-Speechのモデルを利用した、アプリケーションの、説明を、以下に記載します。<br /><br />

                    <img src='/blog/system12.png' alt='twelvethsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、GPT4に問い合わせるプロンプトを、フロントで入力して、それをバックエンド側で、転送します。<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    フロントからの、(POST)APIリクエストを受けて、Apigatewayに転送します。<br /><br />

                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">OpenAIのAPI向けのLambda関数</span><br /><br />
                    まずは、GPT4に、ユーザー入力のテキストに対して、レスポンスを生成してもらいます。<br /><br />
                    GPT4が生成したテキストを、OpenAの、Text-to-Speechのモデルに送信して、テキストから、音声に変換してもらいます。<br /><br />

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

export default BlogArticle14;
