import React from 'react';
import './7.css';

const BlogArticle7 = () => {

    const pythonCode = `
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
import json


def create_dataset(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].str.strip()
    df = df[df['text'] != '']
    return df['text']


def build_vocab(data):
    counter = Counter()
    for text in data:
        counter.update(text.split())
    vocab = {word: i+1 for i, (word, _) in enumerate(counter.items())}
    vocab_size = len(vocab) + 1
    return vocab, vocab_size


class TextDataset(Dataset):
    def __init__(self, texts, vocab):
        self.texts = [text.split() for text in texts]
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded_text = [self.vocab.get(word, 0) for word in self.texts[idx]]
        return torch.tensor(encoded_text[:-1], dtype=torch.long), torch.tensor(encoded_text[1:], dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        x = self.embedding(x)
        if h0 is None and c0 is None:
            h0 = torch.zeros(self.num_layers, batch_size,
                             self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size,
                             self.hidden_size).to(x.device)
        out, (h0, c0) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out, (h0, c0)


def collate_batch(batch):
    input_list, target_list = zip(*batch)
    input_tensor = pad_sequence(input_list, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(
        target_list, batch_first=True, padding_value=0)
    return input_tensor, target_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])

    args, _ = parser.parse_known_args()
    data_dir = args.data_dir

    train_data_path = os.path.join(data_dir, 'cleaned_dataset.csv')
    print(f"Training data path: {train_data_path}")

    df = create_dataset(train_data_path)

    train_texts, _ = train_test_split(df, test_size=0.2)
    vocab, vocab_size = build_vocab(train_texts)
    vocab_path = os.path.join(args.model_dir, 'vocab.json')

    with open(vocab_path, 'w') as vocab_file:
        json.dump(vocab, vocab_file)

    model = LSTMModel(vocab_size, 100, 512, 2, vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    dataset = TextDataset(train_texts, vocab)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    num_epochs = 15

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}")
        for inputs, targets in data_loader:
            print("Running a batch...")
            optimizer.zero_grad()

            outputs, _ = model(inputs)

            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            print(f"Batch loss: {loss.item()}")

    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

    `;

    return (
        <div className='App'>
            <img src='/blog/20240113_12_25_0.png' alt='seventh' className='header-image' />
            <div className='page-title'>
                <h1>Emotional Analysis/Text Generation for dealing with inquiry/claim</h1>
            </div>
            <div className='page-date'>
                <p>2023/05/28</p>
            </div>
            <div className='paragraph'>
                <p>
                    Emotional Analysis/Text Generation for dealing with inquiry/claim<br /><br />

                    このアプリケーションの、具体的な構築手順は、以下の通りです。<br /><br />

                    <img src='/blog/system7.png' alt='fifthsystem' className='system-image' /><br /><br />

                    <span className="highlight">データセットの作成と前処理</span><br /><br />
                    トレーニングに使用する、データセットを作成します。テキストデータを含む、CSVファイルから、データを読み込んでいます。<br /><br />
                    データの前処理として、空白や、不要な文字を、削除し、必要なデータだけを、抽出しています。<br /><br />


                    <span className="highlight">語彙(Vocab)の構築</span><br /><br />
                    テキストデータから、単語の集合を、作成します。各単語には、一意のインデックスが、割り当てられます。<br /><br />

                    <span className="highlight">データセットとデータローダーの準備</span><br /><br />
                    前処理された、テキストデータを用いて、カスタムのデータセットクラスを、作成します。<br /><br />
                    データローダーを使用して、ミニバッチ単位で、データをトレーニングモデルに、供給するための準備をします。<br /><br />

                    <span className="highlight">モデルの定義</span><br /><br />
                    LSTM(Long Short-Term Memory)を使用した、リカレントニューラルネットワーク(RNN)モデルを、定義します。<br /><br />
                    テキストデータの、次の単語を予測するために、使用されます。<br /><br />

                    <span className="highlight">モデルのトレーニング</span><br /><br />
                    損失関数と、最適化手法を定義し、モデルをトレーニングします。<br /><br />
                    各エポックで、モデルは入力テキストに基づいて、次の単語を予測し、予測された単語と、実際の単語との間の、誤差を最小限に抑えるように、学習します。<br /><br />

                    <span className="highlight">モデルの保存と読み込み</span><br /><br />
                    model_fn 関数は、保存されたモデルを読み込んで、推論の準備をします。<br /><br />

                    <span className="highlight">テキスト生成のための推論</span><br /><br />
                    保存されたモデルを使用して、新しいテキストデータに基づいて、テキストを生成します。<br /><br />
                    モデルは、現在の単語に基づいて、次の単語を予測し、これを繰り返して、テキストを生成します。<br /><br />

                    ★以下は検証動画です

                    <br /><br />  <video className="system-video" controls>
                        <source src="/blog/claim.mp4" type="video/mp4" />
                        ご利用のブラウザはこのビデオをサポートしていません。
                    </video>

                    <br /><br />以下は、忘備録として、バックエンドサービスである、Sagemaker側で実装した、コードの説明を記載します。<br /><br />

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

export default BlogArticle7;
