import React from 'react';
import './3.css';

const BlogArticle3 = () => {
    const dummyEpoch = 1;
    const dummyEpochs = 10;
    const dummyLoss = 0.05;
    return (
        <div className='App'>
            <img src='/blog/20231004_18_26_0.png' alt='third' className='header-image' />
            <div className='page-title'>
                <h1>歌詞生成Webアプリケーション(Sagemaker)</h1>
            </div>
            <div className='page-date'>
                <p>2023/01/14</p>
            </div>
            <div className='paragraph'>
                <p>
                    歌詞生成アプリケーションを作りました。<br /><br />
                    URL:http://neilaeden.com/music/<br /><br />
                    ※アーティストの歌詞を、参考に作ったので、あくまで個人利用(実験用)アプリケーションとなります<br /><br />

                    基本的に、推論サーバーは、止めています。理由は、常に、動かしておくのは、コストがかかりすぎる為です。<br /><br />

                    このアプリケーションの、各コンポーネントの役割と、処理の流れは、以下の通りです。<br /><br />

                    <span className="highlight">【構成】</span><br /><br />
                    EC2(React/Nodejs) == API Gateway == Lambda == Sagemaker<br /><br />

                    <img src='/blog/system2.png' alt='fourth' className='system-image' /><br /><br />

                    <span className="highlight">★フロントエンド(React)</span><br /><br />
                    ユーザーインターフェース: Reactを使用して、ユーザーインターフェースを提供します。テキストエリア、ボタン、その他のUI要素が含まれます。<br /><br />
                    ユーザー入力の受付: ユーザーがテキストボックスに、歌詞を入力し、生成ボタンをクリックすると、その入力値が取得されます。<br /><br />
                    APIリクエストの送信: ユーザーが生成ボタンをクリックすると、入力されたテキストを含むリクエストがAPI Gateway経由で、バックエンドに送信されます。<br /><br />
                    レスポンスの表示: バックエンドからの、レスポンスを受け取り、画面上に、歌詞を表示します。<br /><br />

                    <span className="highlight">★バックエンド</span>(Node.js)
                    APIリクエストの処理: Node.jsをベースにしたサーバーが、APIリクエストを受け取ります。<br /><br />
                    データの処理と応答: 受け取ったデータを処理し、必要に応じて、データベースや他のサービスとのやり取りを行います。<br /><br />
                    レスポンスの生成: 処理結果に基づいて、レスポンスを生成し、フロントエンドに返送します。<br /><br />

                    <span className="highlight">★API Gateway</span><br /><br />
                    リクエストの中継: フロントエンドからの、リクエストを受け取り、バックエンドサービス(Lambda)に、ルーティングします。<br /><br />

                    <span className="highlight">★AWS Lambda(Python)</span><br /><br />
                    リクエストの処理: API Gatewayからのリクエストを受け取り、必要な処理(データの形式変換や、バリデーション)を行います。<br /><br />
                    SageMakerとの連携: 処理したデータを、SageMakerのエンドポイントに送信し、歌詞生成のための、機械学習モデルを呼び出します。<br /><br />
                    レスポンスの生成: SageMakerからの応答を受け取り、適切な形式で、フロントエンドに返送します。<br /><br />

                    <span className="highlight">★Amazon SageMaker</span><br /><br />
                    機械学習モデルのホスティング: トレーニング済みの機械学習モデル(LSTMベースのモデル)をデプロイし、リアルタイムでの、推論リクエストに応答します。<br /><br />
                    推論: 入力されたテキストデータに基づいて、歌詞を生成し、その結果をLambda関数、及び、フロントエンドに返します。<br /><br />

                    <span className="highlight">★推論処理の概要</span><br /><br />
                    リクエストの受信: SageMakerエンドポイントは、Lambdaからの推論リクエストを、受け取ります。このリクエストには、生成する歌詞のための、テキストが含まれています。<br /><br />

                    モデルのロード: 推論スクリプト(inference.py)における、model_fn関数は、SageMakerに保存されたモデルの、パラメータをロードします。この関数は、モデルを<br /><br />
                    初期化し、学習済みのパラメータを、モデルにロードする役割を持ちます。<br /><br />

                    データの前処理: input_fn関数は、リクエストボディから送信されたデータを処理して、モデルに適した形式に変換します。この場合、テキストデータをトークン化し、<br /><br />
                    それらのトークンを、モデルが理解できる数値に、変換します。<br /><br />

                    推論の実行: predict_fn関数は、処理されたデータをモデルに渡し、歌詞を生成します。この関数は、モデルが新しい歌詞の、次の単語を予測するために、<br /><br />
                    使用されます。<br /><br />

                    レスポンスの生成: output_fn関数は、モデルからの出力を、適切なレスポンス形式(JSON)に変換し、Lambdaに返送します。<br /><br />

                    <span className="highlight">★コードの概要</span><br /><br />
                    モデルの定義 (LSTMNetクラス):<br /><br />
                    LSTMNetは、LSTMベースのニューラルネットワークで、embeddingレイヤーは、単語を、密なベクトル表現に変換し、lstmレイヤーは、テキストデータのシーケンスを、<br /><br />
                    処理し、fc(全結合)レイヤーは、最終的な出力を生成します。<br /><br />

                    モデルのロード (model_fn関数):<br /><br />
                    この関数は、モデルの状態(重みとバイアス)を、読み込みます。vocab_sizeは、モデルが理解できる、単語の総数を、指定します。<br /><br />

                    入力データの処理 (input_fn関数):<br /><br />
                    送信されたテキストを、JSONから読み込み、必要な前処理を行います。<br /><br />

                    推論 (predict_fn関数):<br /><br />
                    この関数は、モデルを使用して、入力テキストに基づいて、新しい歌詞を、生成します。予測された単語のインデックスを取得し、それらを単語に変換して、<br /><br />
                    結果のテキストを生成します。<br /><br />

                    レスポンスの出力 (output_fn関数):<br /><br />
                    推論の結果を、JSON形式に変換して、返します。<br /><br />
                    この流れにより、ユーザーが入力した、テキストに基づいて、新しい歌詞が生成され、フロントエンドに結果が表示されます。<br /><br />

                    <span className="highlight">★LSTMの基本構造:</span><br /><br />

                    LSTMは、ゲートと呼ばれる構造を持ちます。ゲートは、ネットワークが長期記憶を保持し、情報を忘れるタイミングを、決定します。<br /><br />
                    ゲートは、次の3つがあります: 入力ゲート、忘却ゲート、出力ゲート。<br /><br />

                    入力ゲート: 新しい情報を、セル状態に、追加するか決定します。<br /><br />
                    忘却ゲート: 古い情報を、セル状態から、削除するか決定します。<br /><br />
                    出力ゲート: 次の隠れ層への出力を、決定します。<br /><br />

                    <span className="highlight">★推論における数学的処理</span><br /><br />
                    推論時、モデルは、初期テキストの各トークンに対して、以下のLSTMの、計算を行います。<br /><br />
                    生成する、各新しい単語に対して、LSTMは、次の単語の確率分布を、出力します。<br /><br />
                    この確率分布から、最も可能性の高い単語が、選択されます。<br /><br />
                    選択された単語は、次の入力として使用され、このプロセスが、繰り返されます。<br /><br />

                    <span className="highlight">★纏め</span><br /><br />
                    LSTMの推論プロセスでは、与えられたテキストに基づいて、新しいシーケンス(この場合は歌詞)が、生成されます。<br /><br />
                    LSTMのゲート構造により、関連性のある情報が保持され、不要な情報は忘れられるため、より一貫性のあるテキスト生成が、可能になります。<br /><br />

                    <br /><br />以下は、忘備録として、バックエンドサービスである、Sagemaker側で実装した、コードの説明を記載します。

                    <div class="code-box">
                        <code>
                            <p class="code-text white"><span class="code-text blue">import</span> argparse  # コマンドライン引数を解析するためのライブラリをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> os  # OS機能へのアクセスや環境変数の操作を行うためのライブラリをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> json  # JSONのエンコードとデコードを行うためのライブラリをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> torch  # PyTorch、機械学習フレームワークをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> torch.nn as nn  # PyTorchのニューラルネットワーク関連の機能をインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> torch.optim as optim  # PyTorchの最適化アルゴリズムをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">from</span> torch.utils.data <span class="code-text blue">import</span> DataLoader, Dataset  # データローディングのためのクラスをインポートします。</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">from</span> sklearn.preprocessing <span class="code-text blue">import</span> LabelEncoder  # ラベルのエンコーディングを行うためのライブラリをインポートします。</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">def</span> parse_args():  # コマンドライン引数を解析する関数を定義します。</p><br /><br />
                            <p class="code-text white">    parser = argparse.ArgumentParser()  # 引数パーサーを作成します。</p><br /><br />
                            <p class="code-text white">    parser.add_argument('--epochs', type=int, default=20)  # エポック数の引数を定義します。</p><br /><br />
                            <p class="code-text white">    parser.add_argument('--batch_size', type=int, default=32)  # バッチサイズの引数を定義します。</p><br /><br />
                            <p class="code-text white">    parser.add_argument('--embedding_dim', type=int, default=128)  # 埋め込み次元の引数を定義します。</p><br /><br />
                            <p class="code-text white">    parser.add_argument('--hidden_dim', type=int, default=256)  # 隠れ層の次元数の引数を定義します。</p><br /><br />
                            <p class="code-text white">    parser.add_argument('--num_layers', type=int, default=2)  # LSTM層の数の引数を定義します。</p><br /><br />
                            <p class="code-text white">    parser.add_argument('--seq_length', type=int, default=10)  # シーケンス長の引数を定義します。</p><br /><br />
                            <p class="code-text white">    parser.add_argument('--lr', type=float, default=0.0005)  # 学習率の引数を定義します。</p><br /><br />
                            <p class="code-text white">    parser.add_argument('--model_dir', type=str)  # モデルディレクトリの引数を定義します。</p><br /><br />
                            <p class="code-text white">    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])  # データディレクトリの引数を定義します。</p><br /><br />
                            <p class="code-text white">    <span class="code-text blue">return</span> parser.parse_args()  # 解析された引数を返します。</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">class</span> LSTMNet(nn.Module):  # LSTMネットワークのクラスを定義します。</p><br /><br />
                            <p class="code-text white">    <span class="code-text orange">def</span> __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):  # クラスの初期化関数。</p><br /><br />
                            <p class="code-text white">        super(LSTMNet, self).__init__()  # 親クラスのコンストラクタを呼び出します。</p><br /><br />
                            <p class="code-text white">        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 埋め込み層を定義します。</p><br /><br />
                            <p class="code-text white">        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)  # LSTM層を定義します。</p><br /><br />
                            <p class="code-text white">        self.fc = nn.Linear(hidden_dim, vocab_size)  # 出力層を定義します。</p><br /><br />
                            <p class="code-text white">    <span class="code-text orange">def</span> forward(self, x):  # ネットワークの順伝播を定義します。</p><br /><br />
                            <p class="code-text white">        x = self.embedding(x)  # 入力データを埋め込み層に通します。</p><br /><br />
                            <p class="code-text white">        lstm_out, _ = self.lstm(x)  # LSTM層にデータを通します。</p><br /><br />
                            <p class="code-text white">        out = self.fc(lstm_out)  # 出力層で最終的な出力を計算します。</p><br /><br />
                            <p class="code-text white">        <span class="code-text blue">return</span> out  # 最終出力を返します。</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">class</span> LyricsDataset(Dataset):  # 歌詞データセットのクラスを定義します。</p><br /><br />
                            <p class="code-text white">    <span class="code-text orange">def</span> __init__(self, lyrics, seq_length):  # クラスの初期化関数。</p><br /><br />
                            <p class="code-text white">        self.lyrics = lyrics  # 歌詞データを保存します。</p><br /><br />
                            <p class="code-text white">        self.seq_length = seq_length  # シーケンスの長さを設定します。</p><br /><br />
                            <p class="code-text white">    <span class="code-text orange">def</span> __len__(self):  # データセットの長さを返します。</p><br /><br />
                            <p class="code-text white">        <span class="code-text blue">return</span> len(self.lyrics) - self.seq_length  # 実際の長さを計算します。</p><br /><br />
                            <p class="code-text white">    <span class="code-text orange">def</span> __getitem__(self, index):  # 特定のインデックスのデータを返します。</p><br /><br />
                            <p class="code-text white">        inputs = torch.tensor(self.lyrics[index:index+self.seq_length], dtype=torch.long)  # 入力データを取得します。</p><br /><br />
                            <p class="code-text white">        targets = torch.tensor(self.lyrics[index+1:index+self.seq_length+1], dtype=torch.long)  # 目標データを取得します。</p><br /><br />
                            <p class="code-text white">        <span class="code-text blue">return</span> inputs, targets  # 入力と目標のペアを返します。</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">def</span> load_and_preprocess_data(filepath):  # ファイルからデータを読み込み、前処理する関数。</p><br /><br />
                            <p class="code-text white">    <span class="code-text blue">with</span> open(filepath, 'r', encoding='utf-8') <span class="code-text blue">as</span> file:  # ファイルを読み込みます。</p><br /><br />
                            <p class="code-text white">        text = file.read()  # ファイルの内容を読み込みます。</p><br /><br />
                            <p class="code-text white">    text = text.lower().split()  # テキストを小文字に変換し、単語に分割します。</p><br /><br />
                            <p class="code-text white">    vocab = set(text)  # 単語のユニークなセットを作成します。</p><br /><br />
                            <p class="code-text white">    vocab_size = len(vocab)  # ボキャブラリのサイズを計算します。</p><br /><br />
                            <p className="code-text white">
                                {"word_to_index = {word: i "}
                                <span className="code-text blue">for</span>
                                {" i, word "}
                                <span className="code-text blue">in</span>
                                {" enumerate(vocab)}  // 単語からインデックスへのマッピングを作成します。"}
                            </p><br /><br />
                            <p className="code-text white">
                                {"index_to_word = {i: word "}
                                <span className="code-text blue">for</span>
                                {" i, word "}
                                <span className="code-text blue">in</span>
                                {" enumerate(vocab)}  // インデックスから単語へのマッピングを作成します。"}
                            </p><br /><br />
                            <p class="code-text white">    encoded_text = [word_to_index[word] <span class="code-text blue">for</span> word <span class="code-text blue">in</span> text]  # テキストをエンコードします。</p><br /><br />
                            <p class="code-text white">    <span class="code-text blue">return</span> encoded_text, word_to_index, index_to_word  # エンコードされたテキストとマッピングを返します。</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">def</span> main(args):  # メイン関数。</p><br /><br />
                            <p class="code-text white">    data, vocab_dict, vocab_list = load_and_preprocess_data(</p><br /><br />
                            <p class="code-text white">        os.path.join(args.data_dir, 'preprocessed_lyrics.txt'))  # データの読み込みと前処理。</p><br /><br />
                            <p class="code-text white">    dataset = LyricsDataset(data, args.seq_length)  # データセットの作成。</p><br /><br />
                            <p class="code-text white">    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  # データローダーの作成。</p><br /><br />
                            <p class="code-text white">    vocab_size = len(vocab_dict)  # ボキャブラリのサイズ。</p><br /><br />
                            <p class="code-text white">    model = LSTMNet(vocab_size, args.embedding_dim,</p><br /><br />
                            <p class="code-text white">                    args.hidden_dim, args.num_layers)  # モデルのインスタンス化。</p><br /><br />
                            <p class="code-text white">    criterion = nn.CrossEntropyLoss()  # 損失関数の定義。</p><br /><br />
                            <p class="code-text white">    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # オプティマイザの定義。</p><br /><br />

                            <p class="code-text white">    <span class="code-text blue">for</span> epoch <span class="code-text blue">in</span> range(args.epochs):  # トレーニングループ。</p><br /><br />
                            <p class="code-text white">        <span class="code-text blue">for</span> inputs, targets <span class="code-text blue">in</span> data_loader:  # バッチごとのループ。</p><br /><br />
                            <p class="code-text white">            optimizer.zero_grad()  # 勾配を0に初期化。</p><br /><br />
                            <p class="code-text white">            outputs = model(inputs)  # モデルの出力を取得。</p><br /><br />
                            <p class="code-text white">            outputs = outputs.view(-1, vocab_size)  # 出力を適切な形状に変形。</p><br /><br />
                            <p class="code-text white">            loss = criterion(outputs, targets.view(-1))  # 損失を計算。</p><br /><br />
                            <p class="code-text white">            loss.backward()  # 逆伝播。</p><br /><br />
                            <p class="code-text white">            optimizer.step()  # パラメータ更新。</p><br /><br />
                            <p className="code-text white"><br /><br />
                                {`Epoch ${dummyEpoch}/${dummyEpochs}, Loss: ${dummyLoss}`}<br /><br />
                            </p><br /><br />

                            <p class="code-text white">    <span class="code-text blue">if</span> args.model_dir <span class="code-text blue">is</span> None:</p><br /><br />
                            <p class="code-text white">        model_dir = os.environ.get('SM_MODEL_DIR', '.')  # モデルディレクトリの指定。</p><br /><br />
                            <p class="code-text white">    <span class="code-text blue">else</span>:</p><br /><br />
                            <p class="code-text white">        model_dir = args.model_dir</p><br /><br />
                            <p class="code-text white">    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))  # モデルの保存。</p><br /><br />
                            <p class="code-text white">    save_vocab_data(args.model_dir, vocab_size, vocab_dict, vocab_list)  # ボキャブラリデータの保存。</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">def</span> save_vocab_data(model_dir, vocab_size, vocab_dict, vocab_list):  # ボキャブラリデータを保存する関数。</p><br /><br />
                            <p class="code-text white">    <span class="code-text blue">if</span> model_dir <span class="code-text blue">is</span> None:</p><br /><br />
                            <p class="code-text white">        model_dir = os.environ.get('SM_MODEL_DIR', '.')  # モデルディレクトリの指定。</p><br /><br />
                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_size.json'), 'w', encoding='utf-8') as f:</p><br /><br />
                            <p class="code-text white">        json.dump(vocab_size, f)  # ボキャブラリサイズの保存。</p><br /><br />
                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_dict.json'), 'w', encoding='utf-8') as f:</p><br /><br />
                            <p class="code-text white">        json.dump(vocab_dict, f, ensure_ascii=False)  # ボキャブラリ辞書の保存。</p><br /><br />
                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_list.json'), 'w', encoding='utf-8') as f:</p><br /><br />
                            <p class="code-text white">        json.dump(vocab_list, f, ensure_ascii=False)  # ボキャブラリリストの保存。</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">def</span> load_vocab_data(model_dir):  # ボキャブラリデータを読み込む関数。</p><br /><br />
                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_size.json'), 'r') as f:</p><br /><br />
                            <p class="code-text white">        vocab_size = json.load(f)  # ボキャブラリサイズの読み込み。</p><br /><br />
                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_dict.json'), 'r') as f:</p><br /><br />
                            <p class="code-text white">        vocab_dict = json.load(f)  # ボキャブラリ辞書の読み込み。</p><br /><br />
                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_list.json'), 'r') as f:</p><br /><br />
                            <p class="code-text white">        vocab_list = json.load(f)  # ボキャブラリリストの読み込み。</p><br /><br />

                            <p class="code-text white"><span class="code-text blue">if</span> __name__ <span class="code-text blue">==</span> '__main__':</p><br /><br />
                            <p class="code-text white">    args = parse_args()  # コマンドライン引数の解析。</p><br /><br />
                            <p class="code-text white">    main(args)  # メイン関数の実行。</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">class</span> LSTMNet(nn.Module):  # LSTMネットワークのクラス定義。</p><br /><br />
                            <p class="code-text white">    <span class="code-text orange">def</span> __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):</p><br /><br />
                            <p class="code-text white">        super(LSTMNet, self).__init__()</p><br /><br />
                            <p class="code-text white">        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 埋め込みレイヤーの定義。</p><br /><br />
                            <p class="code-text white">        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)  # LSTMレイヤーの定義。</p><br /><br />
                            <p class="code-text white">        self.fc = nn.Linear(hidden_dim, vocab_size)  # 全結合レイヤーの定義。</p><br /><br />

                            <p class="code-text white">    <span class="code-text orange">def</span> forward(self, x):</p><br /><br />
                            <p class="code-text white">        x = self.embedding(x)  # 埋め込みレイヤーを通過。</p><br /><br />
                            <p class="code-text white">        lstm_out, _ = self.lstm(x)  # LSTMレイヤーを通過。</p><br /><br />
                            <p class="code-text white">        out = self.fc(lstm_out)  # 全結合レイヤーを通過し出力を得る。</p><br /><br />
                            <p class="code-text white">        return out</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">class</span> LyricsDataset(Dataset):  # 歌詞データセットのクラス定義。</p><br /><br />
                            <p class="code-text white">    <span class="code-text orange">def</span> __init__(self, lyrics, seq_length):</p><br /><br />
                            <p class="code-text white">        self.lyrics = lyrics  # 歌詞データ。</p><br /><br />
                            <p class="code-text white">        self.seq_length = seq_length  # シーケンス長。</p><br /><br />

                            <p class="code-text white">    <span class="code-text orange">def</span> __len__(self):</p><br /><br />
                            <p class="code-text white">        return len(self.lyrics) - self.seq_length  # データセットの長さ。</p><br /><br />

                            <p class="code-text white">    <span class="code-text orange">def</span> __getitem__(self, index):</p><br /><br />
                            <p class="code-text white">        inputs = torch.tensor(self.lyrics[index:index+self.seq_length], dtype=torch.long)  # 入力データ。</p><br /><br />
                            <p class="code-text white">        targets = torch.tensor(self.lyrics[index+1:index+self.seq_length+1], dtype=torch.long)  # 目標データ。</p><br /><br />
                            <p class="code-text white">        return inputs, targets</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">def</span> load_and_preprocess_data(filepath):  # データの読み込みと前処理関数。</p><br /><br />
                            <p class="code-text white">    with open(filepath, 'r', encoding='utf-8') as file:</p><br /><br />
                            <p class="code-text white">        text = file.read()  # ファイルからテキストを読み込む。</p><br /><br />

                            <p class="code-text white">    text = text.lower().split()  # テキストを小文字にして分割。</p><br /><br />
                            <p class="code-text white">    vocab = set(text)  # ボキャブラリセットの生成。</p><br /><br />
                            <p class="code-text white">    vocab_size = len(vocab)  # ボキャブラリのサイズ。</p><br /><br />

                            <p className="code-text white">
                                {"word_to_index = {word: i "}
                                <span className="code-text blue">for</span>
                                {" i, word "}
                                <span className="code-text blue">in</span>
                                {" enumerate(vocab)}  // 単語からインデックスへのマッピング。"}
                            </p><br /><br />
                            <p className="code-text white">
                                {"index_to_word = {i: word "}
                                <span className="code-text blue">for</span>
                                {" i, word "}
                                <span className="code-text blue">in</span>
                                {" enumerate(vocab)}  // インデックスから単語へのマッピング。"}
                            </p><br /><br />
                            <p class="code-text white">    encoded_text = [word_to_index[word] for word in text]  # テキストをエンコード。</p><br /><br />
                            <p class="code-text white"><span class="code-text orange">def</span> main(args):  # メイン関数。</p><br /><br />
                            <p class="code-text white">    data, vocab_dict, vocab_list = load_and_preprocess_data(os.path.join(args.data_dir, 'preprocessed_lyrics.txt'))  # データの読み込みと前処理。</p><br /><br />
                            <p class="code-text white">    dataset = LyricsDataset(data, args.seq_length)  # LyricsDatasetのインスタンス作成。</p><br /><br />
                            <p class="code-text white">    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  # DataLoaderの設定。</p><br /><br />

                            <p class="code-text white">    vocab_size = len(vocab_dict)  # ボキャブラリサイズの計算。</p><br /><br />
                            <p class="code-text white">    model = LSTMNet(vocab_size, args.embedding_dim, args.hidden_dim, args.num_layers)  # LSTMモデルのインスタンス作成。</p><br /><br />
                            <p class="code-text white">    criterion = nn.CrossEntropyLoss()  # 損失関数の設定。</p><br /><br />
                            <p class="code-text white">    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 最適化関数の設定。</p><br /><br />

                            <p class="code-text white">    for epoch in range(args.epochs):  # エポック数だけ繰り返す。</p><br /><br />
                            <p class="code-text white">        for inputs, targets in data_loader:  # データローダーからデータを取得。</p><br /><br />
                            <p class="code-text white">            optimizer.zero_grad()  # オプティマイザーの勾配をゼロに設定。</p><br /><br />
                            <p class="code-text white">            outputs = model(inputs)  # モデルから出力を取得。</p><br /><br />
                            <p class="code-text white">            outputs = outputs.view(-1, vocab_size)  # 出力を適切な形状に変形。</p><br /><br />
                            <p class="code-text white">            loss = criterion(outputs, targets.view(-1))  # 損失の計算。</p><br /><br />
                            <p class="code-text white">            loss.backward()  # 逆伝播を実行。</p><br /><br />
                            <p class="code-text white">            optimizer.step()  # オプティマイザーを更新。</p><br /><br />

                            <br /><br /><p className="code-text white">
                                <br /><br />{`Epoch ${dummyEpoch}/${dummyEpochs}, Loss: ${dummyLoss}`}
                            </p><br /><br />

                            <p class="code-text white">    if args.model_dir is None:  # モデルディレクトリのチェック。</p><br /><br />
                            <p class="code-text white">        model_dir = os.environ.get('SM_MODEL_DIR', '.')  # デフォルトのモデルディレクトリを取得。</p><br /><br />
                            <p class="code-text white">    else:</p><br /><br />
                            <p class="code-text white">        model_dir = args.model_dir</p><br /><br />

                            <p class="code-text white">    model_dir = os.environ.get('SM_MODEL_DIR', args.model_dir)</p><br /><br />
                            <p class="code-text white">    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))  # モデルを保存。</p><br /><br />
                            <p class="code-text white">    save_vocab_data(args.model_dir, vocab_size, vocab_dict, vocab_list)  # ボキャブラリデータを保存。</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">def</span> save_vocab_data(model_dir, vocab_size, vocab_dict, vocab_list):  # ボキャブラリデータを保存する関数。</p><br /><br />
                            <p class="code-text white">    if model_dir is None:</p><br /><br />
                            <p class="code-text white">        model_dir = os.environ.get('SM_MODEL_DIR', '.')</p><br /><br />

                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_size.json'), 'w', encoding='utf-8') as f:</p><br /><br />
                            <p class="code-text white">        json.dump(vocab_size, f)  # ボキャブラリサイズを保存。</p><br /><br />

                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_dict.json'), 'w', encoding='utf-8') as f:</p><br /><br />
                            <p class="code-text white">        json.dump(vocab_dict, f, ensure_ascii=False)  # ボキャブラリディクショナリを保存。</p><br /><br />

                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_list.json'), 'w', encoding='utf-8') as f:</p><br /><br />
                            <p class="code-text white">        json.dump(vocab_list, f, ensure_ascii=False)  # ボキャブラリリストを保存。</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">def</span> load_vocab_data(model_dir):  # ボキャブラリデータを読み込む関数。</p><br /><br />
                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_size.json'), 'r') as f:</p><br /><br />
                            <p class="code-text white">        vocab_size = json.load(f)  # ボキャブラリサイズを読み込み。</p><br /><br />

                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_dict.json'), 'r') as f:</p><br /><br />
                            <p class="code-text white">        vocab_dict = json.load(f)  # ボキャブラリディクショナリを読み込み。</p><br /><br />

                            <p class="code-text white">    with open(os.path.join(model_dir, 'vocab_list.json'), 'r') as f:</p><br /><br />
                            <p class="code-text white">        vocab_list = json.load(f)  # ボキャブラリリストを読み込み。</p><br /><br />

                            <p class="code-text white">    return vocab_size, vocab_dict, vocab_list</p><br /><br />

                            <p class="code-text white"><span class="code-text blue">if</span> __name__ == '__main__':  # メイン関数の実行。</p><br /><br />
                            <p class="code-text white">    args = parse_args()  # 引数の解析。</p><br /><br />
                            <p class="code-text white">    main(args)  # メイン関数の実行。</p><br /><br />
                        </code>
                    </div>
                </p>
            </div >
        </div >
    );
};

export default BlogArticle3;
