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

                    <span className="highlight">6. getEmotionText と getEmotionSentence 関数 (App.js内)</span><br /><br />
                    これらの関数は、感情レベルに基づいて対応するテキストを返します。getEmotionText はスライダーのラベルを提供し、getEmotionSentence はユーザーの感情を表す文章を生成します。<br /><br />

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

                    <br /><br />以下は、忘備録として、バックエンドサービスである、Sagemaker側で実装した、コードの説明を記載します。<br /><br />

                    <div class="code-box">
                        <code>
                            <p class="code-text white"><span class="code-text blue">import</span> os オペレーティングシステムの機能やファイルシステムの操作にアクセスするためのモジュールをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> joblib パイソンの機械学習モデルや他の大きなデータ構造を効率的に保存・ロードするためのユーティリティモジュールをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> numpy as np 数値計算を行うためのNumPyライブラリをインポートし、以後npというエイリアスで参照します。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">from</span> sklearn.feature_extraction.text import TfidfVectorizer:Scikit-learn ライブラリからテキストデータを数値的な特徴ベクトルに変換するTF-IDFベクトル化器をインポートします。</p><br /><br />
                            <p class="code-text white">import json JSONデータの処理(エンコード・デコード)を行うためのモジュールをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="highlight-text">def</span> model_fn<span class="code-text orange">(model_dir)</span>:モデルディレクトリパスを引数として受け取り、モデルとベクトル化器をロードする関数を定義します。</p><br /><br />
                            <p class="code-text white">model = joblib.load(os.path.join(model_dir, 'model.joblib')): 指定されたモデルディレクトリからmodel.joblibファイルをロードして、model変数に割り当てます。</p><br /><br />
                            <p class="code-text white">vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib')): 同様に、vectorizer.joblibファイルをロードして、vectorizer変数に割り当てます。</p><br /><br />
                            <p class="code-text white">return model, vectorizer: ロードされたモデルとベクトル化器を戻り値として返します。</p ><br /><br />
                            <p class="code-text white"><span class="highlight-text">def</span> input_fn(request_body, request_content_type):: リクエストの本文とコンテンツタイプを引数として受け取り、リクエストデータを処理する関数を定義します。</p ><br /><br />
                            <p class="code-text white">data = np.array([request_body]): リクエストの本文をNumPy配列に変換します。</p ><br /><br />
                            <p class="code-text white">return data: 変換されたデータを戻り値として返します。</p ><br /><br />
                            <p class="code-text white"><span class="highlight-text">def</span> predict_fn(input_data, model):: 入力データとモデルを引数として受け取り、予測を行う関数を定義します。</p ><br /><br />
                            <p class="code-text white">model, vectorizer = model: 引数で受け取ったモデルからモデルとベクトル化器を展開して取り出します。</p ><br /><br />
                            <p class="code-text white">input_data_tfidf = vectorizer.transform(input_data): 入力データをベクトル化器でTF - IDFベクトルに変換します。</p ><br /><br />
                            <p class="code-text white">predictions = model.predict(input_data_tfidf): 変換されたデータをモデルで予測し、予測結果をpredictionsに割り当てます。</p ><br /><br />
                            <p class="code-text white">labels = ['negative' if pred == 0 else 'positive' for pred in predictions]: 予測結果をラベル（'negative' または 'positive'）に変換します。</p ><br /><br />
                            <p class="code-text white"><span class="highlight-text">def</span>  output_fn(prediction, content_type):: 予測結果とコンテンツタイプを引数として受け取り、適切なフォーマットで結果を返す関数を定義します。</p ><br /><br />
                            <p class="code-text white">return json.dumps(prediction): コンテンツタイプがJSONの場合、予測結果をJSON形式にエンコードして返します。</p ><br /><br />
                            <p class="code-text white">elif content_type == "text/plain":: コンテンツタイプがプレーンテキストの場合、予測結果を文字列に変換して返します。</p ><br /><br />
                            <p class="code-text white">raise ValueError<span class="code-text orange">("Unsupported content type: { }".format(content_type)):</span> サポートされていないコンテンツタイプが指定された場合、エラーを発生させます。</p ><br /><br />
                        </code >
                    </div >
                </p >
            </div >
        </div >
    );
};

export default BlogArticle4;
