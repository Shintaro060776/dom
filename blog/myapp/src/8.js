import React, { useState } from 'react';
import './8.css';

const BlogArticle8 = () => {
    const [image_url] = useState("https://example.com/image.png");

    return (
        <div className='App'>
            <img src='/blog/20240114_12_00_0.png' alt='eighth' className='header-image' />
            <div className='page-title'>
                <h1>Dalle</h1>
            </div>
            <div className='page-date'>
                <p>2023/06/25</p>
            </div>
            <div className='paragraph'>
                <p>
                    Dalle<br /><br />

                    OpenAIのAPIを利用して、(Dalleにて)画像を生成されています。<br /><br />

                    <img src='/blog/system8.png' alt='eightthsystem' className='system-image' /><br /><br />

                    注意点としては、Lambda ランタイムごとに、PATH 変数に /opt ディレクトリ内の特定のフォルダが含まれます。レイヤー .zip ファイルアーカイブに同じフォルダ構造を定義すると、関数コードはパスを指定しなくても、レイヤーコンテンツにアクセスできます。
                    ・Python
                    – パス：python、python/lib/python3.9/site-packages<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/dalle.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video><br /><br />

                    <br /><br />以下は、忘備録として、Lambdaのコードの説明を記載します。<br /><br />

                    <div class="code-box">
                        <code>
                            <p class="code-text white"><span class="code-text blue">import</span> os</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> json</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> requests</p><br /><br />
                            <p class="code-text white">from openai <span class="code-text blue">import</span> OpenAI</p><br /><br />


                            <p class="code-text white"><span class="highlight-text">def</span> lambda_handler(event, context):</p><br /><br />
                            <p class="code-text white">try:</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">api_key = os.environ['OPENAI_API_KEY']</span></p><br /><br />
                            <p class="code-text white">slack_webhook_url = os.environ['SLACK_WEBHOOK_URL']</p><br /><br />
                            <p class="code-text white">client = OpenAI(api_key=api_key)</p><br /><br />

                            <p class="code-text white">body = json.loads(event['body'])</p><br /><br />
                            <p class="code-text white">prompt = body['prompt']</p><br /><br />
                            <p class="code-text white">print(f"Received prompt: {prompt}")</p><br /><br />
                            <p class="code-text white">except Exception as e:</p><br /><br />
                            <p class="code-text white"><span class="highlight-text">print("Error in initializing and getting prompt:", str(e))</span></p><br /><br />
                            <p class="code-text white">{'return {`statusCode`: 500, `body`: json.dumps({`error`: str(e)})}'}</p><br /><br />

                            <p class="code-text white">try:</p><br /><br />
                            <p class="code-text white">response = client.images.generate(</p><br /><br />
                            <p class="code-text white">model="dall-e-3",</p><br /><br />
                            <p class="code-text white">prompt=prompt,</p><br /><br />
                            <p class="code-text white">size="1024x1024",</p><br /><br />
                            <p class="code-text white">quality="standard",</p><br /><br />
                            <p class="code-text white">n=1,</p><br /><br />
                            <p class="code-text white">)</p><br /><br />
                            <p class="code-text white">image_url = response.data[0].url</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">print(f"Generated image URL: {image_url}")</span></p><br /><br />
                            <p class="code-text white">except Exception as e:</p><br /><br />
                            <p class="code-text white">print("Error in generating image:", str(e))</p><br /><br />
                            <p class="code-text white">{'return {`statusCode`: 500, `body`: json.dumps({`error`: str(e)})}'}</p><br /><br />

                            <p class="code-text white">try:</p><br /><br />
                            <p class="code-text white">{'slack_message = {'}</p><br /><br />
                            <p class="code-text white">    "blocks": [</p><br /><br />
                            <p class="code-text white">{"{"}</p><br /><br />
                            <p class="code-text white">    "type": "section",</p><br /><br />
                            <p class="code-text white">"text": {"{"}</p><br /><br />
                            <p class="code-text white">    "type": "mrkdwn",</p><br /><br />
                            <p class="code-text white">"text": f"Generated Image: {prompt}"</p><br /><br />
                            <p class="code-text white">{"},"}</p><br /><br />
                            <p class="code-text white">"accessory": {"{"}</p><br /><br />
                            <p class="code-text white">    "type": "image",</p><br /><br />
                            <p class="code-text white">"image_url": image_url,</p><br /><br />
                            <p class="code-text white">"alt_text": "Generated image"</p><br /><br />
                            <p class="code-text white">{"},"}</p><br /><br />
                            <p class="code-text white">{"},"}</p><br /><br />
                            <p class="code-text white">]</p><br /><br />
                            <p class="code-text white">{"},"}</p><br /><br />
                            <p class="code-text white"><span class="highlight-text">requests.post(slack_webhook_url, json=slack_message)</span></p><br /><br />
                            <p class="code-text white">except Exception as e:</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">print("Error in sending message to Slack:", str(e))</span></p><br /><br />

                            <p class="code-text white">return {"{"}</p><br /><br />
                            <p class="code-text white">    'statusCode': 200,</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">'body': json.dumps({"{"}'imageUrl': image_url{"}"})</span></p><br /><br />
                            <p class="code-text white">{"},"}</p><br /><br />

                        </code >
                    </div >
                </p >
            </div >
        </div >
    );
};

export default BlogArticle8;
