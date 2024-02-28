import React from 'react';
import './5.css';

const BlogArticle5 = () => {
    return (
        <div className='App'>
            <img src='/blog/20231126_12_19_0.png' alt='fifth' className='header-image' />
            <div className='page-title'>
                <h1>Realtime Chatbot Emotional Analysis</h1>
            </div>
            <div className='page-date'>
                <p>2023/03/27</p>
            </div>
            <div className='paragraph'>
                <p>
                    Realtime Chatbot Emotional Analysis<br /><br />

                    このアプリケーションの、各コンポーネントの役割と、処理の流れは、以下の通りです。<br /><br />

                    <img src='/blog/system5.png' alt='fifthsystem' className='system-image' /><br /><br />

                    <span className="highlight">React (フロントエンド):</span><br /><br />
                    ユーザーがテキストを入力し、サーバーに送信するための入力フィールドを提供します。<br /><br />
                    サーバーからの応答(AIによる返答や感情分析の結果)を表示します。<br /><br />


                    <span className="highlight">Node.js</span><br /><br />
                    Node.jsはAPIリクエストを受け取り、処理します。<br /><br />

                    <span className="highlight">API Gateway</span><br /><br />
                    フロントエンド(Reactアプリ)とバックエンド(Lambda関数)間の通信を管理します。<br /><br />
                    エンドポイントとして機能し、HTTPリクエストを受け取って適切なLambda関数に転送します。<br /><br />

                    <span className="highlight">Lambda</span><br /><br />
                    API Gatewayからのリクエストに基づいて、必要な処理(例えば、SageMakerへの感情分析リクエストの送信や、OpenAIへのテキスト生成リクエストの処理)を行います。<br /><br />

                    <span className="highlight">SageMaker</span><br /><br />
                    機械学習モデルを使用して、ユーザーからの入力テキストの感情を分析します。<br /><br />

                    <span className="highlight">DynamoDB</span><br /><br />
                    ユーザーの入力、感情分析の結果、AIの応答などのデータを保存する。<br /><br />

                    ★以下は、Davinciをモデルとして使った場合の検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/video.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video><br /><br />

                    ★以下は、gpt-3.5-turboをモデルとして使った場合の検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/video1.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video>

                    <br /><br />以下は、忘備録として、バックエンドサービスである、Sagemaker側で実装した、コードの説明を記載します。<br /><br />

                    <div class="code-box">
                        <code>
                            <p class="code-text white"><span class="code-text blue">import</span> argparse: コマンドライン引数を解析するための標準ライブラリ</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> os: オペレーティングシステム関連の機能を提供する標準ライブラリ</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> pandas as <span class="code-text blue">pd</span>: データ分析に使用されるPandasライブラリをインポート</p><br /><br />
                            <p class="code-text white">from sklearn.model_selection <span class="code-text blue">import</span> train_test_split: データセットをトレーニングセットとテストセットに分割する関数</p><br /><br />
                            <p class="code-text white">from sklearn.feature_extraction.text <span class="code-text blue">import</span> TfidfVectorizer: テキストデータをTF-IDF特徴量に変換するためのクラス</p><br /><br />
                            <p class="code-text white">from sklearn.svm <span class="code-text blue">import</span> SVC: サポートベクターマシン分類器</p><br /><br />
                            <p class="code-text white">from sklearn.metrics <span class="code-text blue">import</span> classification_report, accuracy_score: 分類結果の評価に使用される関数</p><br /><br />
                            <p class="code-text white">from sklearn.pipeline <span class="code-text blue">import</span> Pipeline: 前処理とモデルフィッティングのステップを一つのワークフローに結合する</p><br /><br />
                            <p class="code-text white">from sklearn.model_selection <span class="code-text blue">import</span> GridSearchCV: パラメータの最適化に使用されるグリッドサーチのクラス</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> joblib: モデルを保存するためのライブラリ</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> boto3: AWSサービスとの連携を可能にするPython SDK</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> logging: ロギングを行うための標準ライブラリ</p><br /><br />

                            <p class="code-text white">logging.basicConfig(level=logging.INFO)ロギングの基本設定を行う</p><br /><br />


                            <p class="code-text white"><span class="highlight-text">def</span> parse_args():</p><br /><br />
                            <p class="code-text white">parser = argparse.ArgumentParser(): コマンドライン引数を解析するための関数を定義</p>

                            <p class="code-text white">parser.add_argument<span class="code-text orange">('--max_df', type=float, default=0.5)</span>: コマンドライン引数の設定</p><br /><br />
                            <p class="code-text white">parser.add_argument<span class="code-text orange">('--C', type=float, default=1.0)</span>: コマンドライン引数を解析して変数に割り当て</p><br /><br />

                            <p class="code-text white">return parser.parse_known_args()</p><br /><br />


                            <p class="code-text white">if __name__ == '__main__':</p><br /><br />
                            <p class="code-text white">args, _ = parse_args()</p><br /><br />

                            <p class="code-text white">input_data_path = os.environ['SM_CHANNEL_TRAIN']: 環境変数からトレーニングデータのパスを取得</p><br /><br />
                            <p class="code-text white">output_model_path = os.environ['SM_MODEL_DIR']: モデルの出力パスを環境変数から取得</p><br /><br />

                            <p class="code-text white">s3 = boto3.client('s3')S3サービスのクライアントを作成</p><br /><br />
                            <p class="code-text white">bucket = 'realtime20090317'</p><br /><br />
                            <p class="code-text white">file_key = 'preprocessed.csv'</p><br /><br />
                            <p class="code-text white">s3.download_file(bucket, file_key, 'preprocessed.csv'): S3からファイルをダウンロード</p><br /><br />

                            <p class="code-text white">df = pd.read_csv('preprocessed.csv'): ダウンロードしたCSVファイルをPandasデータフレームとして読み込み</p><br /><br />
                            <p class="code-text white">df = df.dropna(subset=['clean_tweet']): 欠損値を持つ行を削除</p><br /><br />

                            <p class="code-text white">X = df['clean_tweet']: 特徴量とターゲット変数を定義</p><br /><br />
                            <p class="code-text white">y = df['Sentiment']</p><br /><br />

                            <p class="code-text white">logging.info("データを分離しています...")</p><br /><br />
                            <p class="code-text white">X_train, X_test, y_train, y_test = train_test_split(: データセットをトレーニングセットとテストセットに分割</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">X, y, test_size=0.2, random_state=42)</span></p><br /><br />

                            <p class="code-text white">pipeline = Pipeline([: パイプラインを定義(TF-IDF変換とSVM分類器を含む)</p><br /><br />
                            <p class="code-text white">('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=args.max_df)),</p><br /><br />
                            <p class="code-text white">('clf', SVC(C=args.C)),</p><br /><br />
                            <p class="code-text white">])</p><br /><br />

                            <p class="code-text white"><span class="highlight-text">pipeline.fit(X_train, y_train)</span>: パイプラインをトレーニングデータにフィットさせる</p><br /><br />

                            <p class="code-text white">parameters = {"{"} : グリッドサーチに使用するパラメータの設定</p><br /><br />
                            <p class="code-text white">   'tfidf__max_df': (0.5, 0.75, 1.0),</p><br /><br />
                            <p class="code-text white">'clf__C': [1, 10, 100],</p><br /><br />
                            <p class="code-text white">{"}"}</p><br /><br />

                            <p class="code-text white">logging.info("グリッドサーチを開始します...")</p><br /><br />
                            <p class="code-text white">grid_search = GridSearchCV(: グリッドサーチを初期化</p><br /><br />
                            <p class="code-text white">pipeline, parameters, cv=5, n_jobs=-1, verbose=3)</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">grid_search.fit(X_train, y_train):</span> トレーニングデータでグリッドサーチを実行</p><br /><br />

                            <p class="code-text white">logging.info("予測と評価を行います")</p><br /><br />
                            <p class="code-text white">predictions = grid_search.predict(X_test): テストデータで予測を行う</p><br /><br />
                            <p class="code-text white"><span class="highlight-text">print("Best Parameters:", grid_search.best_params_)</span>: 最適なパラメータを表示</p><br /><br />
                            <p class="code-text white"><span class="highlight-text">print("Accuracy:", accuracy_score(y_test, predictions))</span>: 精度を表示</p><br /><br />
                            <p class="code-text white"><span class="highlight-text">print(classification_report(y_test, predictions))</span>: 分類レポートを表示</p><br /><br />

                            <p class="code-text white">joblib.dump(grid_search.best_estimator_, os.path.join(: トレーニングされたモデルをファイルに保存</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">output_model_path, 'model.joblib'))</span></p><br /><br />

                        </code >
                    </div >
                </p >
            </div >
        </div >
    );
};

export default BlogArticle5;
