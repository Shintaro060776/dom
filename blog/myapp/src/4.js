import React from 'react';
import './4.css';

const BlogArticle4 = () => {
    return (
        <div className='App'>
            <img src='/blog/20231018_07_37_0.png' alt='fourth' className='header-image' />
            <div className='page-title'>
                <h1>感情分析アプリケーション</h1>
            </div>
            <div className='page-date'>
                <p>2023/02/19</p>
            </div>
            <div className='paragraph'>
                <p>
                    感情分析アプリケーション<br /><br />

                    このアプリケーションの、各コンポーネントの役割と、処理の流れは、以下の通りです。<br /><br />

                    <img src='/blog/system3.png' alt='fourth' className='system-image' /><br /><br />

                    <span className="highlight">★フロントエンド(React)</span><br /><br />
                    ユーザーインターフェース: Reactを使用して、ユーザーインターフェースを提供します。テキストエリア、ボタン、その他のUI要素が含まれます。<br /><br />
                    ユーザー入力の受付: ユーザーがテキストボックスに、歌詞を入力し、生成ボタンをクリックすると、その入力値が取得されます。<br /><br />
                    APIリクエストの送信: ユーザーが生成ボタンをクリックすると、入力されたテキストを含むリクエストがAPI Gateway経由で、バックエンドに送信されます。<br /><br />
                    レスポンスの表示: バックエンドからの、レスポンスを受け取り、画面上に、歌詞を表示します。<br /><br />


                    <span className="highlight">1. App コンポーネント (App.js)</span><br /><br />
                    Appコンポーネントは、アプリケーションの主要なコンポーネントであり、他のコンポーネントを組み合わせて、全体のUIを形成します。状態(state)を管理し、ユーザー入力と、APIリクエストを処理します。主要な機能は、以下の通りです<br /><br />

                    ユーザーのテキスト入力を受け取る。<br /><br />
                    感情レベルスライダーの状態を管理する。<br /><br />
                    「Inquiry」ボタンがクリックされた時に、OpenAI APIへの、リクエストを送信する。<br /><br />
                    OpenAIからのレスポンスを表示する。<br /><br />

                    <span className="highlight">2. Header コンポーネント (Header.js)</span><br /><br />
                    Header コンポーネントは、アプリケーションのヘッダー部分を表示します。アプリケーションの名前やタイトルが含まれています。<br /><br />

                    <span className="highlight">3. Footer コンポーネント (Footer.js)</span><br /><br />
                    Footer コンポーネントは、アプリケーションのフッター部分を表示します。著作権情報などが含まれています。<br /><br />

                    <span className="highlight">4. EmotionSlider コンポーネント (App.js内)</span><br /><br />
                    EmotionSlider コンポーネントは、感情レベルを調整するためのスライダーです。ユーザーはこのスライダーを使用して、感情レベルを1から5まで調整できます。<br /><br />

                    <span className="highlight">5. ResponseDisplay コンポーネント (App.js内)</span><br /><br />
                    ResponseDisplay コンポーネントは、OpenAIからのレスポンスを表示します。タイプライター風のアニメーションで一文字ずつ表示され、カーソルが点滅します。

                    <span className="highlight">6. getEmotionText と getEmotionSentence 関数 (App.js内)</span>
                    これらの関数は、感情レベルに基づいて対応するテキストを返します。getEmotionText はスライダーのラベルを提供し、getEmotionSentence はユーザーの感情を表す文章を生成します。

                    <span className="highlight">7. Node.js</span><br /><br />
                    Node.jsは、このアプリケーションのサーバーサイド環境として機能します。<br /><br />

                    ユーザーからのリクエストを処理し、適切なレスポンスを返す。<br /><br />

                    <span className="highlight">8. API Gateway</span><br /><br />
                    API Gatewayは、クライアントとバックエンドサービス間の「ゲートウェイ」として機能します。このアプリケーションでは、以下のような役割を果たしています<br /><br />

                    ユーザーのHTTPリクエストを受け取り、それをLambda関数にルーティングする。<br /><br />
                    Lambda関数からのレスポンスを受け取り、それをクライアントに返す。<br /><br />

                    <span className="highlight">9. AWS Lambda</span><br /><br />
                    AWS Lambdaは、サーバーレスコンピューティングを提供するサービスです<br /><br />

                    API Gatewayからのリクエストを受け取る。<br /><br />
                    必要に応じて、翻訳や感情分析を行い、OpenAI APIへ問い合わせる。<br /><br />
                    OpenAIからのレスポンスを処理し、結果をAPI Gatewayを通じてクライアントに返す。<br /><br />

                    <span className="highlight">10. Amazon SageMaker</span><br /><br />
                    Amazon SageMakerは、機械学習モデルのトレーニング、デプロイ、管理を簡素化するフルマネージドサービスです。<br /><br />

                    感情分析のための機械学習モデルをトレーニングする。<br /><br />
                    トレーニングされたモデルをSageMakerのエンドポイントとしてデプロイする。<br /><br />
                    AWS Lambdaからのリクエストに応じて、テキストの感情分析を行い、結果をLambdaに返す。<br /><br />

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/3.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video>
                </p>
            </div >
        </div >
    );
};

export default BlogArticle4;
