import React from 'react';
import './10.css';

const BlogArticle10 = () => {

    return (
        <div className='App'>
            <img src='/blog/20240118_07_45_0.png' alt='tenth' className='header-image' />
            <div className='page-title'>
                <h1>Speech-to-Text/GPT4</h1>
            </div>
            <div className='page-date'>
                <p>2023/08/30</p>
            </div>
            <div className='paragraph'>
                <p>
                    Speech-to-Text/GPT4<br /><br />

                    OpenAIのAPIを利用して、音声データを、テキストに変換して、その生成されたテキストを、GPT4で要約しています<br /><br />

                    <img src='/blog/system10.png' alt='tenthsystem' className='system-image' /><br /><br />

                    <span className="highlight">フロントエンド</span><br /><br />
                    ユーザーは、Reactであるフロントにて、音声ファイルを、選択します。<br /><br />
                    音声ファイルは、Reactアプリケーションによって、状態管理されます。<br /><br />
                    「アップロード」ボタンをクリックすると、Reactアプリケーションは、バックエンドのAPI(/api/upload)に、リクエストを送信し、プリサインドURLを取得します。<br /><br />
                    取得したプリサインドURLを使用して、フロントエンドから、直接、S3バケットにファイルを、アップロードします。<br /><br />

                    <span className="highlight">バックエンド</span><br /><br />
                    フロントエンドからのリクエストを受け、Lambda関数が、AWS SDKを使用して、S3バケットに対して、プリサインドURLを生成します。<br /><br />
                    このURLは、特定のファイルを、アップロードするために、一時的に、使用されます。<br /><br />
                    S3バケットに、ファイルがアップロードされると、これがトリガーとなり、OpenAIのAPIを叩く為の、Lambda関数が実行されます。<br /><br />
                    Lambda関数が、S3から音声ファイルを取得します。<br /><br />
                    OpenAIのAPIを使用して、音声ファイルを、テキストに変換します。<br /><br />

                    <span className="highlight">外部サービス(OpenAI)</span><br /><br />
                    Lambda関数は、取得した音声ファイルを、OpenAIのWhisperモデルに、送信します。<br /><br />
                    Whisperモデルは、音声をテキストに変換し、その結果を、Lambda関数に返します。<br /><br />
                    Lambda関数は、変換されたテキストを、OpenAIのモデルである、GPT-4に送信し、要約を行います。<br /><br />
                    GPT-4は、要約されたテキストを返します。<br /><br />

                    <span className="highlight">結果の保存と通知</span><br /><br />
                    Lambda関数は、要約されたテキストを、DynamoDBに保存します。<br /><br />
                    保存された情報は、将来的に、フロントエンドで参照されるために、使用されます。<br /><br />
                    Lambda関数は、変換されたテキストを、OpenAIのモデルである、GPT-4に送信し、要約を行います。<br /><br />
                    このタイミングで、Slackへ通知します。<br /><br />

                    <span className="highlight">フロントエンドへの結果表示</span><br /><br />
                    ユーザーが、「要約を表示」ボタンをクリックすると、Reactアプリケーションは、バックエンドのAPIに、リクエストを送信し、要約されたテキストを取得します。<br /><br />
                    取得した要約されたテキストは、フロントにて、表示されます。<br /><br />

                    <br /><br />★以下は、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/gpt4p.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video><br /><br />

                    <br /><br />以下は、忘備録として、Lambdaのコードの説明を記載します。<br /><br />

                    <div class="code-box">
                        <code>
                        </code >
                    </div >
                </p >
            </div >
        </div >
    );
};

export default BlogArticle9;
