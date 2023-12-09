import React from 'react';
import './2.css';

const BlogArticle2 = () => {
    return (
        <div className='App'>
            <img src='/blog/20231016_12_05_0.png' alt='second' className='header-image' />
            <div className='page-title'>
                <h1>機械学習を用いた実験的Webアプリケーションについて</h1>
            </div>
            <div className='page-date'>
                <p>2023/01/09</p>
            </div>
            <div className='paragraph'>
                <p>
                    実験サイトで作った、機械学習を用いたサービスの概要を、以下に記載します。<br /><br />

                    <span className="highlight">①動くヌコ</span><br /><br />
                    URL : http://neilaeden.com/predict/<br /><br />
                    アプリの概要：ユーザーがフォーム上に、ネコの絵を書いて、その書いた絵を、アニメーション(動かす)させるというWebアプリケーション<br /><br />
                    難易度：★★★★☆<br /><br />
                    雑感 : Web上に、なかなか手書きのネコの画像が無いから、完成度は、そこまで高くないかな。推論処理は動くし、エラーも出ていないから、ま、いいか、という感じ。
                    何故、ネコだけにしたかというと、ネコ以外の学習データ(犬、鳥、魚など)も取り入れると、機械学習(AI)サーバーの学習時間が増大して、費用が高くなるから、限定的にしたという経緯です。スマホで、フォームに絵を書けるけど、書きづらいから、PCのブラウザから、どうぞ

                    <br /><br /><span className="highlight">②Joke GPT</span><br /><br />
                    URL : http://neilaeden.com/natural/<br /><br />
                    アプリの概要 : ChatGPTの劣化版だと思っていて、ユーザーが入力フォーム(テキストエリア)に英単語を入力すると、機械学習(AI)サーバーが、英語でジョークを生成する<br /><br />
                    難易度：★★★★★<br /><br />
                    雑感：日本語でジョーク(シャレ)を生成したかったけど、Web上に、そういう感じのデータが無いから、日本語Versionは断念。しかし、英語だと、そういうデータがあったから、
                    トライしてみた。ただ、AIに学習させたデータ量は、そんなに多くないから、質はぼちぼち。機械学習(AI)サーバーから、レスポンス(ジョーク)が返ってくるから良しとした。

                    <br /><br /><span className="highlight">【その他】</span><br /><br />
                    アフターコロナ対策として、3年前ぐらいから、数学の勉強をしていて、高校の数学から着手したのだけど、これが完全にターニングポイントだったなと思いますね
                    AIは数学的な知見であるとか、知識が非常に重要で、何故ならAIは数式そのものだから。
                    当然、コーディング(プログラミング)スキルとか、環境(インフラとか)構築スキルとか、トラブルシューティング、エラーハンドリングとかも、重要やけど、
                    数学が一番カギを握る印象。とっくに、大学数学(微分積分学、線形代数学など)まで到達していて、来年は、もっとAIを理解する為に、家庭教師に、来てもらおうと思っている

                    <br /><br />あとは、学習させるデータ量が多ければ、当然、これらのwebアプリケーションのクオリティ(所望のリスポンス)は、余裕で上がる

                    <br /><br /><span className="highlight">①動くヌコ</span><br /><br />
                    ★フロントエンド(React)<br /><br />
                    キャンバス描画: ユーザーは、キャンバス上に絵を描けて、マウスやタッチイベントを利用することにより、描画機能を使えます。
                    画像のリサイズ: 描かれた画像は、サーバーに送信する前に、特定のサイズ(256x256ピクセル)に、リサイズします。
                    これは、サーバー側のモデルが、特定のサイズの入力を期待している為。
                    アニメーションの実行: サーバーからの応答に基づいて、特定のアニメーション(口が動く)が実行されます。これは、描かれた画像の内容によって、
                    異なるアニメーションを選択する為。<br /><br />

                    <span className="highlight">★AIについての説明</span><br /><br />
                    このアプリケーションで使用される、AIの主要部分は、画像を解析し、特定のラベルに分類するディープラーニングモデルで、CNN(Convolutional Neural Networks)というモデルは、
                    画像内のパターンや特徴を学習し、それを基に画像を、特定のカテゴリに分類します。このプロセスは、AIは大量のデータと、計算能力を用いて、実行します。

                    <br /><br /><span className="highlight">②Joke GPT</span><br /><br />
                    ★アプリケーションの概要：<br /><br />
                    このWebアプリケーションは、ユーザーインターフェース(フロントエンド)、バックエンドサービス、自然言語処理(NLP)サービスを統合して、ジョークを生成する
                    ロジックを組んでいます。<br /><br />

                    ★推論とジョークの生成<br /><br />
                    モデルのデプロイと推論: トレーニングされたモデルは、Webアプリケーションのバックエンド(AWS SageMaker)でデプロイされ、リアルタイムの推論リクエストに、応答します。
                    生成プロセス: ユーザーからの入力(一部の単語、または、フレーズ)に基づいて、モデルは続く単語を、一つずつ生成します。この過程で、モデルは以前の単語に基づいて、
                    次の単語を予測し、ジョークを形成します。<br /><br />

                    ★アプリケーションの概要<br /><br />
                    このアプリケーションは、ユーザーが描いた画像に基づいて、アニメーションを生成するもので。ユーザーは、キャンバスに絵を描き、その絵がサーバーに送信され、
                    特定のラベル(例えば「笑顔」や「怒り」など)に基づいて、フロント(React)側のコードにより、アニメーションが実行されるというロジックです。
                    以降は、各ポイントでの処理の内容を、記載します。<br /><br />

                    ★フロントエンド(React)<br /><br />
                    キャンバス描画: ユーザーは、キャンバス上に絵を描けて、マウスやタッチイベントを利用することにより、描画機能を使えます。
                    画像のリサイズ: 描かれた画像は、サーバーに送信する前に、特定のサイズ(256x256ピクセル)に、リサイズします。
                    これは、サーバー側のモデルが、特定のサイズの入力を期待している為。
                    アニメーションの実行: サーバーからの応答に基づいて、特定のアニメーション(口が動く)が実行されます。これは、描かれた画像の内容によって、
                    異なるアニメーションを選択する為。<br /><br />

                    ★バックエンド(AIと機械学習)<br /><br />
                    画像処理: サーバーは、ユーザーから受け取った画像を処理し、それをモデルに供給するために、前処理します。
                    モデル予測: 画像は、ディープラーニングモデル(CNN)に供給され、モデルは、画像の内容に基づいて、予測(ラベル)を行います。
                    応答の生成: モデルからの予測結果は、クライアントに送り返され、フロントエンドで適切なアニメーションを、トリガーします。<br /><br />

                    <br /><br />以下は、このシステムの構成となります。

                    <br /><br /><img src='/blog/system1.png' alt='third' className='system-image' /><br /><br />

                    <br /><br />以下は、忘備録として、バックエンドサービスである、Sagemaker側で実装した、コードの説明を記載します。

                    <div class="code-box">
                        <code>
                            <p class="code-text white"><span class="code-text blue">import</span> argparse  # コマンドライン引数解析のためのモジュールをインポートします。</p> <br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> os  # オペレーティングシステムの機能にアクセスするためのモジュールをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> pandas as pd  # データ操作と分析のためのライブラリPandasをpdとしてインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> torch  # PyTorchライブラリをインポートします(ディープラーニング用)。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> torch.nn as nn  # PyTorchのニューラルネットワークモジュールをnnとしてインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">from</span> torch.utils.data import Dataset, DataLoader  # PyTorchのデータセットとデータローダーのクラスをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">from</span> torch.nn.utils.rnn import pad_sequence  # シーケンスデータのパディング（長さを揃える処理）のための関数をインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">from</span> torch.optim import Adam  # PyTorchのAdamオプティマイザーをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">from</span> collections import Counter  # オブジェクトのカウントに使うCounterクラスをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> tarfile  # tarアーカイブの読み書きのためのモジュールをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> re  # 正規表現のためのモジュールをインポートします。</p><br /><br /><br /><br />

                            <p class="code-text white"><span class="code-text orange">def</span> preprocess_text(text):</p><br /><br />
                            <p class="code-text white">text = text.lower()  # テキストを小文字にします。</p><br /><br />
                            <p class="code-text white">text = re.sub(r"[^a-z0-9\s]", '', text)  # 英数字と空白以外の文字を削除します。</p><br /><br />
                            <p class="code-text white">text = re.sub(r"\s+", ' ', text)  # 一つ以上の空白を単一の空白に置換します。</p><br /><br />
                            <p class="code-text white">return text.strip()  # 文字列の前後の空白を削除します。</p><br /><br />
                            <p class="code-text white">class JokesDataset(Dataset):</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">def</span> __init__(self, sequences, labels):</p><br /><br />
                            <p class="code-text white">self.sequences = sequences  # シーケンスデータを属性に設定します。</p><br /><br />
                            <p class="code-text white">self.labels = labels  # ラベルデータを属性に設定します。</p><br /><br /><br /><br />

                            <p class="code-text white"><span class="code-text orange">def</span> __len__(self):</p><br /><br />
                            <p class="code-text white">return len(self.sequences)  # データセットの長さを返します。</p><br /><br /><br /><br />

                            <p class="code-text white"><span class="code-text orange">def</span> __getitem__(self, idx):</p><br /><br />
                            <p class="code-text white">return self.sequences[idx], self.labels[idx]  # インデックスに対応するデータを返します。</p><br /><br />
                            <p class="code-text white">class JokeGeneratorModel(nn.Module):</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">def</span> __init__(self, vocab_size, embed_dim, hidden_dim):</p><br /><br />
                            <p class="code-text white">super().__init__()  # 親クラスのコンストラクタを呼び出します。</p><br /><br />
                            <p class="code-text white">self.embedding = nn.Embedding(vocab_size, embed_dim)  # 埋め込み層を定義します。</p><br /><br />
                            <p class="code-text white">self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)  # LSTM層を定義します。</p><br /><br />
                            <p class="code-text white">self.fc = nn.Linear(hidden_dim, vocab_size)  # 全結合層を定義します。</p><br /><br /><br /><br />

                            <p class="code-text white"><span class="code-text orange">def</span> forward(self, x):</p><br /><br />
                            <p class="code-text white">embedded = self.embedding(x)  # 入力データを埋め込みます。</p><br /><br />
                            <p class="code-text white">lstm_out, _ = self.lstm(embedded)  # LSTM層を通します。</p><br /><br />
                            <p class="code-text white">logits = self.fc(lstm_out)  # 全結合層を通して出力を計算します。</p><br /><br />
                            <p class="code-text white">return logits  # ロジット（未正規化の確率）を返します。</p><br /><br /><br /><br />

                            <p class="code-text white">if __name__ == '__main__':</p><br /><br />
                            <p class="code-text white">parser = argparse.ArgumentParser() # コマンドライン引数を定義します。</p><br /><br />

                            <p class="code-text white">args = parser.parse_known_args()[0] # コマンドライン引数を解析します。</p><br /><br />

                            <p class="code-text white">df = pd.read_csv(os.path.join(args.data_dir, 'normalized_jokes.csv')) # CSVファイルを読み込みます。</p><br /><br />

                            <p class="code-text white">jokes = df['Normalized Joke'].apply(preprocess_text).tolist() # ジョークの前処理を行いリストに変換します。</p><br /><br />
                            <p class="code-text white"># トークンカウント、語彙の構築、ジョークのエンコーディングなどの処理を行います。</p><br /><br />

                            <p class="code-text white">dataset = JokesDataset(sequences, labels) # ジョークデータセットを作成します。</p><br /><br /><br /><br />

                            <p class="code-text white">train_loader = DataLoader(</p><br /><br />
                            <p class="code-text white">dataset, batch_size=args.batch_size, shuffle=True) # データセットからデータローダーを作成します。</p><br /><br /><br /><br />

                            <p class="code-text white">model = JokeGeneratorModel(</p><br /><br />
                            <p class="code-text white">args.vocabulary_size, embed_dim=100, hidden_dim=128) # ジョーク生成モデルをインスタンス化します。</p><br /><br />

                            <p class="code-text white">criterion = nn.CrossEntropyLoss() # 損失関数を定義します。</p><br /><br />

                            <p class="code-text white">optimizer = Adam(model.parameters()) # オプティマイザーを定義します。</p><br /><br />

                            <p class="code-text white">for epoch in range(args.epochs): # 訓練ループを実行します。</p><br /><br />

                            <p class="code-text white">model_save_path = os.path.join(args.model_dir, 'model.pth') # モデルの保存パスを定義します。</p><br /><br />

                            <p class="code-text white">archive_path = os.path.join(args.model_dir, 'model.tar.gz') # アーカイブの保存パスを定義します。</p><br /><br />

                            <p class="code-text white">torch.save(model.state_dict(), model_save_path) # モデルの状態を保存します。</p><br /><br />

                            <p class="code-text white">with tarfile.open(archive_path, mode='w:gz') as archive:</p><br /><br />
                            <p class="code-text white">archive.add(model_save_path, arcname='model.pth') # モデルをtar.gzファイルとしてアーカイブします。</p><br /><br />
                        </code>
                    </div>

                    <br /><br />

                    <br /><br />★AIについての説明<br /><br />
                    このアプリケーションで使用される、AIの主要部分は、画像を解析し、特定のラベルに分類するディープラーニングモデルで、CNN(Convolutional Neural Networks)というモデルは、
                    画像内のパターンや特徴を学習し、それを基に画像を、特定のカテゴリに分類します。このプロセスは、AIは大量のデータと、計算能力を用いて、実行します。
                    <br /><br />
                    ****************************************************************
                    <br /><br />
                    <span className="highlight">②Joke GPT</span><br /><br />
                    ★アプリケーションの概要：<br /><br />
                    このWebアプリケーションは、ユーザーインターフェース(フロントエンド)、バックエンドサービス、自然言語処理(NLP)サービスを統合して、ジョークを生成する
                    ロジックを組んでいます。<br /><br />

                    ★データの前処理<br /><br />
                    テキストの正規化: ジョークのデータセットを収集し、テキストを小文字に変換、特殊文字の削除、単語間の空白の整理などを行って、正規化します。
                    トークン化と語彙の構築: 正規化されたテキストを、単語単位に分割(トークン化)し、データセット全体の単語に基づいて、語彙(単語とインデックスのマッピング)を作成します。
                    <br /><br />
                    ★モデルのトレーニング<br /><br />
                    LSTMモデルの使用: LSTM(Long Short-Term Memory)ネットワークを使用し、ジョークの生成に必要な、シーケンシャルなパターンを学習する。
                    LSTMは時系列データや、テキストデータにおいて、長期的な依存関係を学習するのに適している。
                    シーケンスデータの準備: ジョークの各単語を、語彙のインデックスに変換し、それらをシーケンス(単語の連続)として、モデルに供給します。
                    ターゲットは、次に来る単語を予測すること。<br /><br />

                    ★推論とジョークの生成<br /><br />
                    モデルのデプロイと推論: トレーニングされたモデルは、Webアプリケーションのバックエンド(AWS SageMaker)でデプロイされ、リアルタイムの推論リクエストに、応答します。
                    生成プロセス: ユーザーからの入力(一部の単語、または、フレーズ)に基づいて、モデルは続く単語を、一つずつ生成します。この過程で、モデルは以前の単語に基づいて、
                    次の単語を予測し、ジョークを形成します。<br /><br />

                    ★ポストプロセッシング<br /><br />
                    生成されたテキストの調整: モデルによって生成されたテキストは、ポストプロセッシング(不適切な内容の除去、文法的な調整など)が、必要になる。

                    <br /><br />★ユーザーインターフェースとの統合<br /><br />
                    フロントエンド: Reactを、フロントエンドフレームワークとして使用して、ユーザーが入力したテキストを受け取り、生成されたジョークを表示する。
                    <br /><br />非同期通信: ユーザーからの入力を受け取り、バックエンドの推論サーバーに送信し、応答を受け取って、表示するプロセスは、非同期的に行われます。

                    <br /><br />上記が、このWebアプリケーションにおける、AI的な処理の、ロジックの概要で、ユーザーからの入力に基づいて、笑いのあるテキストを、生成するために、
                    <br /><br />自然言語処理の技術と、機械学習の手法によって、実装しています。
                </p>
            </div>
        </div>
    );
};

export default BlogArticle2;
