import React from 'react';
import './6.css';

const BlogArticle6 = () => {
    return (
        <div className='App'>
            <img src='/blog/20240105_14_01_0.png' alt='sixth' className='header-image' />
            <div className='page-title'>
                <h1>Category Classification/Text Generation</h1>
            </div>
            <div className='page-date'>
                <p>2023/04/29</p>
            </div>
            <div className='paragraph'>
                <p>

                    このアプリケーションは、カテゴリ分類と、テキスト生成の、二つの主要な機能を持っています。<br /><br />

                    <img src='/blog/system6.png' alt='sixthsystem' className='system-image' /><br /><br />

                    <span className="highlight">カテゴリ分類</span><br /><br />
                    この機能は、入力されたテキストを、特定のカテゴリに、分類します。<br /><br />
                    「愛」、「冒険」、「友情」など、さまざまなカテゴリが、考えられます。<br /><br />
                    ユーザーからの入力テキストは、まず、Amazon Translateを使用して、英語に翻訳され、その後、Amazon SageMakerに、デプロイされたカテゴリ分類モデルに、送られます。<br /><br />
                    このモデルは、テキストを分析し、それが属するカテゴリを、特定します。<br /><br />
                    この分類結果は、後のステップ、つまりテキスト生成に使用されます。<br /><br />

                    <span className="highlight">テキスト生成</span><br /><br />
                    カテゴリ分類の結果に基づいて、テキスト生成モデルは、新しいテキストを、生成します。<br /><br />
                    この機能は、特定のカテゴリに基づいて、関連するテキストを、作成するために設計されています。<br /><br />
                    たとえば、カテゴリが、「愛」であれば、愛に関連する言葉や、フレーズを含むテキストが、生成されます。<br /><br />
                    このプロセスでは、再び、Amazon SageMakerに、デプロイされたテキスト生成モデルが、使用され、入力されたカテゴリと、翻訳されたテキストに基づいて、新しいテキストが、生成されます。<br /><br />
                    生成されたテキストは、最終的に、Amazon Translateを使用して、日本語に翻訳され、ユーザーに返されます。<br /><br />

                    ★カテゴリ分離/テキスト生成の、検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/gen.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video>

                    <br /><br />以下は、忘備録として、バックエンドサービスである、Sagemaker側で実装した、コードの説明を記載します。<br /><br />

                    <div class="code-box">
                        <code>
                            <p class="code-text white"><span class="code-text blue">import</span> os</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> argparse</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> torch</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> torch.nn as nn</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> torch.optim as optim</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> numpy as np</p><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> pandas as pd</p ><br /><br />
                            <p class="code-text white">from torch.utils.data <span class="code-text blue">import</span> Dataset, DataLoader</p ><br /><br />
                            <p class="code-text white">from collections <span class="code-text blue">import</span> Counter</p ><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> nltk</p ><br /><br />
                            <p class="code-text white">from nltk.tokenize <span class="code-text blue">import</span> word_tokenize</p ><br /><br />
                            <p class="code-text white">from sklearn.model_selection <span class="code-text blue">import</span> train_test_split</p ><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> logging</p ><br /><br />
                            <p class="code-text white"><span class="code-text blue">import</span> json</p ><br /><br />
                            <p class="code-text white">ライブラリのインポート: PyTorch (深層学習のため)、NumPy、Pandas（データ操作のため）、NLTK（自然言語処理）、argparse（コマンドライン引数処理のため）などが含まれます。</p><br /><br />

                            <p class="code-text white">logging.basicConfig(level = logging.INFO,</p ><br /><br />
                            <p class="code-text white">format='%(asctime)s %(levelname)s: %(message)s')</p><br /><br />

                            <p class="code-text white">try:</p><br /><br />
                            <p class="code-text white">logging.info("Attempting to download NLTK punkt dataset")</p><br /><br />
                            <p class="code-text white">nltk.download('punkt')</p><br /><br />
                            <p class="code-text white">logging.info("Successfully downloaded NLTK punkt dataset")</p><br /><br />
                            <p class="code-text white">except Exception as e:</p><br /><br />
                            <p class="code-text white">logging.error("Error downloading NLTK punkt dataset: %s", e)</p><br /><br />
                            <p class="code-text white">raise e</p><br /><br />

                            <p class="code-text white">class TextDataset(Dataset):</p><br /><br />
                            <p class="code-text white"><span class="highlight-text">def</span> __init__(self, sequences, vocab_size):</p><br /><br />
                            <p class="code-text white">self.sequences = sequences</p><br /><br />
                            <p class="code-text white">self.vocab_size = vocab_size</p><br /><br />

                            <p class="code-text white"><span class="highlight-text">def</span> __len__(self):</p><br /><br />
                            <p class="code-text white">return len(self.sequences)</p><br /><br />
                            <p class="code-text white">TextDatasetクラス: PyTorchのDatasetを継承しており、テキストデータのシーケンスを処理します。これは、モデル訓練のためのデータローダーに使用されます。</p><br /><br />

                            <p class="code-text white"><span class="highlight-text">def</span> __getitem__(self, idx):</p><br /><br />
                            <p class="code-text white">seq = self.sequences[idx]</p><br /><br />
                            <p class="code-text white">input_seq = torch.tensor(seq[:-1], dtype=torch.long)</p><br /><br />
                            <p class="code-text white">target_seq = torch.tensor(seq[1:], dtype=torch.long)</p><br /><br />
                            <p class="code-text white">return input_seq, target_seq</p><br /><br />


                            <p class="code-text white">class LSTMModel(nn.Module):</p><br /><br />
                            <p class="code-text white"><span class="highlight-text">def</span> __init__(self, vocab_size, embedding_dim, hidden_dim):</p><br /><br />
                            <p class="code-text white">super().__init__()</p><br /><br />
                            <p class="code-text white">self.embedding = nn.Embedding(vocab_size, embedding_dim)</p><br /><br />
                            <p class="code-text white">self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)</p><br /><br />
                            <p class="code-text white">self.fc = nn.Linear(hidden_dim, vocab_size)</p><br /><br />

                            <p class="code-text white"><span class="highlight-text">def</span> forward(self, text):</p><br /><br />
                            <p class="code-text white">embedded = self.embedding(text)</p><br /><br />
                            <p class="code-text white">output, _ = self.lstm(embedded)</p><br /><br />
                            <p class="code-text white">prediction = self.fc(output)</p><br /><br />
                            <p class="code-text white">return torch.softmax(prediction, dim=-1)</p><br /><br />
                            <p class="code-text white">LSTMModelクラス: LSTM（Long Short-Term Memory）を用いたニューラルネットワークモデルを定義しています。これは、テキストデータのシーケンスから次の単語を予測するために使用されます。</p>

                            <p class="code-text white"><span class="highlight-text">def</span> preprocess_text(text_data, seq_length):</p><br /><br />
                            <p class="code-text white">tokenized_texts = [word_tokenize(text.lower()) for text in text_data]</p><br /><br />
                            <p class="code-text white">vocab = Counter()</p><br /><br />
                            <p class="code-text white">for text in tokenized_texts:</p><br /><br />
                            <p class="code-text white">vocab.update(text)</p><br /><br />

                            <p class="code-text white">logging.info(f"Built vocab: {vocab}")</p><br /><br />

                            <p class="code-text white">word_to_idx = {"{word: i + 2 for i, word in enumerate(vocab)}"}</p><br /><br />
                            <p class="code-text white">word_to_idx["{"<pad>"}"] = 0</p><br /><br />
                            <p className="code-text white">word_to_idx["{"<unk>"}"] = 1</p><br /><br />

                            <p className="code-text white">idx_to_word = {"{i: word for word, i in word_to_idx.items()}"}</p><br /><br />
                            <p class="code-text white">logging.info(f"Word to index mapping: {word_to_idx}")</p><br /><br />

                            <p class="code-text white">sequences = []</p><br /><br />
                            <p class="code-text white">for text in tokenized_texts:</p><br /><br />
                            <p class="code-text white">encoded_text = [word_to_idx.get(word, 1) for word in text]</p><br /><br />
                            <p class="code-text white">for i in range(len(encoded_text) - seq_length):</p><br /><br />
                            <p class="code-text white">sequences.append(encoded_text[i:i + seq_length + 1])</p><br /><br />

                            <p class="code-text white">logging.info(f"Number of sequences: {len(sequences)}")</p><br /><br />
                            <p class="code-text white">return sequences, word_to_idx, idx_to_word, len(word_to_idx)</p><br /><br />
                            <p class="code-text white">テキストの前処理: preprocess_text 関数は、テキストデータをトークン化し、単語をインデックスにマッピングする処理を行います。</p><br /><br />

                            <p class="code-text white"><span class="highlight-text">def</span> create_category_dataset(df, category, seq_length):</p><br /><br />
                            <p class="code-text white">category_texts = df[df['Category'] == category]['Quote']</p><br /><br />
                            <p class="code-text white">logging.info(</p><br /><br />
                            <p class="code-text white">f"Processing category: {category} with {len(category_texts)} texts")</p ><br /><br />

                            <p class="code-text white">sequences, word_to_idx, idx_to_word, vocab_size = preprocess_text(</p ><br /><br />
                            <p class="code-text white">category_texts, seq_length)</p><br /><br />

                            <p class="code-text white">logging.info(f"Category '{category}' vocab size: {vocab_size}")</p><br /><br />

                            <p class="code-text white">return sequences, word_to_idx, idx_to_word, vocab_size</p><br /><br />
                            <p class="code-text white">カテゴリデータセットの作成: create_category_dataset 関数は、特定のカテゴリに属するテキストデータのシーケンスを生成します。</p><br /><br />

                            <p class="code-text white"><span class="highlight-text">def</span> train_model(sequences, vocab_size, model, batch_size=128, num_epochs=10, model_dir='/opt/ml/model'):</p><br /><br />
                            <p class="code-text white">dataset = TextDataset(sequences, vocab_size)</p><br /><br />
                            <p class="code-text white">train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)</p><br /><br />

                            <p class="code-text white">optimizer = optim.Adam(model.parameters())</p><br /><br />
                            <p class="code-text white">criterion = nn.CrossEntropyLoss()</p><br /><br />

                            <p class="code-text white">for epoch in range(num_epochs):</p><br /><br />
                            <p class="code-text white">model.train()</p><br /><br />
                            <p class="code-text white">total_loss = 0</p><br /><br />

                            <p class="code-text white">for input_seq, target_seq in train_loader:</p><br /><br />
                            <p class="code-text white">optimizer.zero_grad()</p><br /><br />
                            <p class="code-text white">output = model(input_seq)</p><br /><br />

                            <p class="code-text white">logging.info(f"Output shape: {output.shape}")</p><br /><br />
                            <p class="code-text white">logging.info(f"Target sequence shape: {target_seq.shape}")</p><br /><br />
                            <p class="code-text white">logging.info(f"vocab_size being used: {vocab_size}")</p><br /><br />

                            <p class="code-text white">output = output.view(-1, vocab_size)</p><br /><br />

                            <p class="code-text white">loss = criterion(output, target_seq.view(-1))</p><br /><br />
                            <p class="code-text white">loss.backward()</p><br /><br />
                            <p class="code-text white">optimizer.step()</p><br /><br />

                            <p class="code-text white">total_loss += loss.item()</p><br /><br />

                            <p class="code-text white">avg_loss = total_loss / len(train_loader)</p><br /><br />
                            <p class="code-text white">print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")</p><br /><br />

                            <p class="code-text white">model_path = os.path.join(model_dir, 'model.pth')</p><br /><br />
                            <p class="code-text white">torch.save(model.state_dict(), model_path)</p><br /><br />

                            <p class="code-text white">meta_data = {"{"}</p><br /><br />
                            <p class="code-text white">'vocab_size': vocab_size,</p ><br /><br />
                            <p class="code-text white">'embedding_dim': model.embedding.embedding_dim,</p><br /><br />

                            <p class="code-text white">'hidden_dim': model.lstm.hidden_size</p ><br /><br />
                            <p class="code-text white">{"}"}</p><br /><br />

                            <p class="code-text white">with open(os.path.join(model_dir, 'meta.json'), 'w') as meta_file:</p ><br /><br />
                            <p class="code-text white">json.dump(meta_data, meta_file)</p ><br /><br />
                            <p class="code-text white">モデルの訓練: train_model 関数は、定義されたLSTMモデルを使用してテキストデータを訓練します。</p ><br /><br />

                            <p class="code-text white">if __name__ == "__main__":</p ><br /><br />
                            <p class="code-text white">try:</p ><br /><br />
                            <p class="code-text white">parser = argparse.ArgumentParser()</p ><br /><br />

                            <p class="code-text white"><span class="code-text orange">parser.add_argument('--model-dir', type = str,</span></p ><br /><br />
                            <p class="code-text white"><span class="code-text orange">default=os.environ['SM_MODEL_DIR'])</span></p ><br /><br />
                            <p class="code-text white">parser.add_argument('--data-dir', type = str,</p ><br /><br />
                            <p class="code-text white">default=os.environ['SM_CHANNEL_TRAIN'])</p ><br /><br />

                            <p class="code-text white">args, _ = parser.parse_known_args()</p ><br /><br />

                            <p class="code-text white">logging.info("Starting model training")</p ><br /><br />

                            <p class="code-text white">csv_file_path = os.path.join(</p ><br /><br />
                            <p class="code-text white">args.data_dir, 'preprocessed_new_and_new.csv')</p><br /><br />
                            <p class="code-text white">logging.info(f"Loading data from {csv_file_path}")</p><br /><br />
                            <p class="code-text white">df = pd.read_csv(csv_file_path)</p><br /><br />
                            <p class="code-text white">categories = df['Category'].unique()</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">all_texts = df['Quote']</span></p><br /><br />
                            <p class="code-text white">_, word_to_idx, _, vocab_size = preprocess_text(</p><br /><br />
                            <p class="code-text white">all_texts, seq_length=10)</p><br /><br />

                            <p class="code-text white">with open(os.path.join(args.model_dir, 'vocab.json'), 'w') as vocab_file:</p><br /><br />
                            <p class="code-text white">json.dump(word_to_idx, vocab_file)</p><br /><br />

                            <p class="code-text white">all_models_state = { }</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">model = LSTMModel(vocab_size, 50, 100)</span></p><br /><br />
                            <p class="code-text white">logging.info(f"Model output size: {model.fc.out_features}")</p><br /><br />

                            <p class="code-text white">for category in categories:</p><br /><br />
                            <p class="code-text white">if not df[df['Category'] == category]['Quote'].empty:</p><br /><br />
                            <p class="code-text white">logging.info(f"Processing category: {category}")</p><br /><br />
                            <p class="code-text white">sequences, _, _, _ = create_category_dataset(</p><br /><br />
                            <p class="code-text white">df, category, seq_length=10)</p><br /><br />

                            <p class="code-text white">logging.info(</p><br /><br />
                            <p class="code-text white">f"Length of word_to_idx for category '{category}' : {len(word_to_idx)}")</p><br /><br />
                            <p class="code-text white">logging.info(f"Using vocab_size: {vocab_size} for training")</p><br /><br />

                            <p class="code-text white">if sequences:</p><br /><br />
                            <p class="code-text white">logging.info(f"Training model for category: {category}")</p><br /><br />
                            <p class="code-text white">train_model(sequences, vocab_size, model, batch_size=128,</p><br /><br />
                            <p class="code-text white">num_epochs=10, model_dir=args.model_dir)</p><br /><br />

                            <p class="code-text white">all_models_state[category] = model.state_dict()</p><br /><br />
                            <p class="code-text white">logging.info(</p><br /><br />
                            <p class="code-text white">f"Completed training for category: {category}")</p><br /><br />

                            <p class="code-text white"><span class="code-text orange">model_save_path = os.path.join(args.model_dir, 'all_models.tar')</span></p><br /><br />
                            <p class="code-text white">torch.save(all_models_state, model_save_path)</p><br /><br />
                            <p class="code-text white">logging.info(f"Model saved to {model_save_path}")</p><br /><br />

                            <p class="code-text white">except Exception as e:</p><br /><br />
                            <p class="code-text white">logging.error("Error in training: %s", e)</p><br /><br />
                            <p class="code-text white">raise e</p><br /><br />
                            <p class="code-text white">メインの実行ブロック: このスクリプトのメイン部分では、コマンドライン引数を処理し、データを読み込み、全てのカテゴリに対してモデルの訓練を行い、訓練されたモデルを保存します。</p><br /><br />

                        </code >
                    </div >
                </p >
            </div >
        </div >
    );
};

export default BlogArticle6;
