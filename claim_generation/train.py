import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


def create_dataset(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
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

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


def collate_batch(batch):
    input_list, target_list = zip(*batch)
    input_tensor = pad_sequence(input_list, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(
        target_list, batch_first=True, padding_value=0)
    return input_tensor, target_tensor


df = create_dataset("cleaned_dataset.csv")
train_texts, _ = train_test_split(df, test_size=0.2)
vocab, vocab_size = build_vocab(train_texts)

embed_size = 100
hidden_size = 512
num_layers = 2
output_size = vocab_size
learning_rate = 0.001
batch_size = 128
num_epochs = 30

model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

dataset = TextDataset(train_texts, vocab)
data_loader = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, collate_fn=collate_batch)

if __name__ == "__main__":
    try:
        print("Training started...")
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch + 1}")
            for inputs, targets in data_loader:
                print("Running a batch...")
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, vocab_size),
                                 targets.view(-1))
                loss.backward()
                optimizer.step()
                print(f"Batch loss: {loss.item()}")

            print(f"Epoch [{epoch+1}/{num_epochs}] completed.")

        torch.save(model.state_dict(), "lstm_model.pth")
        print("Training completed and model saved.")
    except Exception as e:
        print(f"An error occurred: {e}")
