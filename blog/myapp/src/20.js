import React from 'react';
import './20.css';

const BlogArticle20 = () => {

    const pythonCode = `
    import json
    import boto3
    from datetime import datetime
    import openai
    import os
    
    dynamodb = boto3.resource('dynamodb')
    translate = boto3.client('translate')
    table = dynamodb.Table('SmokeFreeUserData')
    
    def lambda_handler(event, context):
        print("Received event:", event)
        try:
            data = json.loads(event['body'])
            print("Parsed data:", data)
            user_id = data.get('user_id', 'default_user_id')
            smoke_free_days = data['smoke_free_days']
            cigarettes_not_smoked = data['cigarettes_not_smoked']
            money_saved = data['money_saved']
    
            timestamp = datetime.now().isoformat()
            item = {
                'user_id': user_id,
                'timestamp': timestamp,
                'smoke_free_days': smoke_free_days,
                'cigarettes_not_smoked': cigarettes_not_smoked,
                'money_saved': money_saved
            }
    
            table.put_item(Item=item)
    
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            user_prompt = f"How can I stay motivated after being smoke-free for {smoke_free_days} days, not smoking {cigarettes_not_smoked} cigarettes, and saving {money_saved} dollars?"
    
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}
                ]
            )
    
            ai_response = response.choices[0].message.content
    
            translated_response = translate.translate_text(
                Text=ai_response,
                SourceLanguageCode='en',
                TargetLanguageCode='ja'
            )['TranslatedText']
    
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Data updated successfully',
                    'ai_response': translated_response
                })
            }
    
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Error processing request',
                    'message': str(e)
                })
            }
        `;

    return (
        <div className='App'>
            <img src='/blog/smokefree.png' alt='twenteenth' className='header-image' />
            <div className='page-title'>
                <h1>SmokeFree</h1>
            </div>
            <div className='page-date'>
                <p>2024/5/7</p>
            </div>
            <div className='paragraph'>
                <p>
                    SmokeFree<br /><br />

                    今回は、行動変容(禁煙)を促す、アプリケーションについて、以下に記載します。<br /><br />

                    <img src='/blog/system20.png' alt='twenteenthsystem' className='system-image' /><br /><br />

                    <br /><br /><span className="highlight">フロント(React)</span><br /><br />
                    ユーザーが、禁煙の進捗を、テキストボックスに入力します<br /><br />

                    <br /><br /><span className="highlight">バックエンド(Nodejs)</span><br /><br />
                    フロントからの、(POST)APIリクエストを受けて、Apigatewayに転送します。<br /><br />

                    <br /><br /><span className="highlight">Apigateway</span><br /><br />
                    Nodejsからの、APIリクエストを受けて、Lambdaに転送します。<br /><br />

                    <br /><br /><span className="highlight">Dynamodbにプロンプト情報を登録する/OpenAIに問い合わせるLambda関数</span><br /><br />
                    上記で、ユーザーが入力する、禁煙の進捗を、Dynamodbに登録して、その進捗状況を、Open AIの、GPT4に問い合わせます<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/smokefree.mp4" type="video/mp4" />
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

export default BlogArticle20;
