import React from 'react';
import './9.css';

const BlogArticle9 = () => {
    const user_input = "xxxxxx";
    const ai_response = "xxxxxx";
    const e = "xxxxxx";


    return (
        <div className='App'>
            <img src='/blog/20240115_10_28_0.png' alt='nineth' className='header-image' />
            <div className='page-title'>
                <h1>GPT4</h1>
            </div>
            <div className='page-date'>
                <p>2023/07/29</p>
            </div>
            <div className='paragraph'>
                <p>
                    GPT4<br /><br />

                    OpenAIのGPT4の、APIを利用して、高品質のテキストを、生成しています<br /><br />

                    <img src='/blog/system9.png' alt='ninethsystem' className='system-image' /><br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/gpt4p.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video><br /><br />

                    <br /><br />以下は、忘備録として、Lambdaのコードの説明を記載します。<br /><br />

                    <div class="code-box">
                        <code>
                            <p class="code-text white"><span class="code-text blue">import</span> json</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> logging</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> openai</p><br /><br />
                            <p class="code-text white">from openai <span class="code-text blue">import</span> OpenAI</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> os</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> requests</p><br /><br />

                            <p class="code-text white">logger = logging.getLogger()</p><br /><br />
                            <p class="code-text white">logger.setLevel(logging.INFO)</p><br /><br />


                            <p class="code-text white"><span class="highlight-text">def</span> send_slack_notification(user_input, ai_response):</p><br /><br />
                            <p class="code-text white">webhook_url = os.environ.get("SLACK_WEBHOOK_URL")</p><br /><br />
                            <p class="code-text white">message = f"User Input: {user_input}\nAI Response: {ai_response}"</p><br /><br />
                            <p class="code-text white">slack_data = &#123;'text': message&#125;</p><br /><br />
                            <p class="code-text white">response = requests.post(webhook_url, json=slack_data)</p><br /><br />
                            <p class="code-text white">return response</p><br /><br />


                            <p class="code-text white"><span class="highlight-text">def</span> lambda_handler(event, context):</p><br /><br />
                            <p class="code-text white">try:</p><br /><br />
                            <p class="code-text white">client = OpenAI(</p><br /><br />
                            <p class="code-text white">api_key=os.environ.get("OPENAI_API_KEY")</p><br /><br />
                            <p class="code-text white">)</p><br /><br />

                            <p class="code-text white">body = json.loads(event["body"])</p><br /><br />
                            <p class="code-text white">user_input = body.get("message", "")</p><br /><br />

                            <p class="code-text white">logger.info(f"User Input: {user_input}")</p><br /><br />

                            <p class="code-text white">response = client.chat.completions.create(</p><br /><br />
                            <p class="code-text white">model="gpt-4",</p><br /><br />
                            <p class="code-text white">messages=[</p><br /><br />
                            <p class="code-text white">&#123;"role": "system", "content": "You are a helpful assistant."&#125;,</p><br /><br />
                            <p class="code-text white">&#123;"role": "user", "content": `{user_input}`&#125;</p><br /><br />
                            <p class="code-text white">]</p><br /><br />
                            <p class="code-text white">)</p><br /><br />

                            <p class="code-text white">ai_response = response.choices[0].message.content</p><br /><br />
                            <p class="code-text white">logger.info(f"AI Response: {ai_response}")</p><br /><br />

                            <p class="code-text white">send_slack_notification(user_input, ai_response)</p><br /><br />

                            <p class="code-text white">return &#123;</p><br /><br />
                            <p class="code-text white">    "statusCode": 200,</p><br /><br />
                            <p class="code-text white">"body": json.dumps(&#123;"response": ai_response&#125;)</p><br /><br />
                            <p class="code-text white">&#125;</p><br /><br />

                            <p class="code-text white">except Exception as e:</p><br /><br />
                            <p class="code-text white">logger.error(f"Error occurred: {String(e)}")</p><br /><br />
                            <p class="code-text white">send_slack_notification("Error occurred", String(e))</p><br /><br />

                            <p class="code-text white">return &#123;</p><br /><br />
                            <p class="code-text white">    "statusCode": 500,</p><br /><br />
                            <p class="code-text white">"body": json.dumps(&#123;"error": "Internal Server Error"&#125;)</p><br /><br />
                            <p class="code-text white">&#125;</p><br /><br />

                        </code >
                    </div >
                </p >
            </div >
        </div >
    );
};

export default BlogArticle9;
