import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import logging
import json


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

try:
    logging.info("Attempting to download NLTK punkt dataset")
    nltk.download('punkt')
    logging.info("Successfully downloaded NLTK punkt dataset")
except Exception as e:
    logging.error("Error downloading NLTK punkt dataset: %s", e)
    raise e


class TextDataset(Dataset):
    def __init__(self, sequences, vocab_size):
        self.sequences = sequences
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.lstm(embedded)
        prediction = self.fc(output)
        return torch.softmax(prediction, dim=-1)


def preprocess_text(text_data, seq_length):
    tokenized_texts = [word_tokenize(text.lower()) for text in text_data]
    vocab = Counter()
    for text in tokenized_texts:
        vocab.update(text)

    logging.info(f"Built vocab: {vocab}")

    word_to_idx = {word: i + 2 for i, word in enumerate(vocab)}
    word_to_idx["<pad>"] = 0
    word_to_idx["<unk>"] = 1

    idx_to_word = {i: word for word, i in word_to_idx.items()}
    logging.info(f"Word to index mapping: {word_to_idx}")

    sequences = []
    for text in tokenized_texts:
        encoded_text = [word_to_idx.get(word, 1) for word in text]
        for i in range(len(encoded_text) - seq_length):
            sequences.append(encoded_text[i:i + seq_length + 1])

    logging.info(f"Number of sequences: {len(sequences)}")
    return sequences, word_to_idx, idx_to_word, len(word_to_idx)


def create_category_dataset(df, category, seq_length):
    category_texts = df[df['Category'] == category]['Quote']
    logging.info(
        f"Processing category: {category} with {len(category_texts)} texts")

    sequences, word_to_idx, idx_to_word, vocab_size = preprocess_text(
        category_texts, seq_length)

    logging.info(f"Category '{category}' vocab size: {vocab_size}")

    return sequences, word_to_idx, idx_to_word, vocab_size


def train_model(sequences, vocab_size, model, batch_size=128, num_epochs=10, model_dir='/opt/ml/model'):
    dataset = TextDataset(sequences, vocab_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for input_seq, target_seq in train_loader:
            optimizer.zero_grad()
            output = model(input_seq)

            logging.info(f"Output shape: {output.shape}")
            logging.info(f"Target sequence shape: {target_seq.shape}")
            logging.info(f"vocab_size being used: {vocab_size}")

            output = output.view(-1, vocab_size)

            loss = criterion(output, target_seq.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

    meta_data = {
        'vocab_size': vocab_size,
        'embedding_dim': model.embedding.embedding_dim,
        'hidden_dim': model.lstm.hidden_size
    }

    with open(os.path.join(model_dir, 'meta.json'), 'w') as meta_file:
        json.dump(meta_data, meta_file)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument('--model-dir', type=str,
                            default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--data-dir', type=str,
                            default=os.environ['SM_CHANNEL_TRAIN'])

        args, _ = parser.parse_known_args()

        logging.info("Starting model training")

        csv_file_path = os.path.join(
            args.data_dir, 'preprocessed_new_and_new.csv')
        logging.info(f"Loading data from {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        categories = df['Category'].unique()

        all_texts = df['Quote']
        _, word_to_idx, _, vocab_size = preprocess_text(
            all_texts, seq_length=10)

        with open(os.path.join(args.model_dir, 'vocab.json'), 'w') as vocab_file:
            json.dump(word_to_idx, vocab_file)

        all_models_state = {}

        model = LSTMModel(vocab_size, 50, 100)
        logging.info(f"Model output size: {model.fc.out_features}")

        for category in categories:
            if not df[df['Category'] == category]['Quote'].empty:
                logging.info(f"Processing category: {category}")
                sequences, _, _, _ = create_category_dataset(
                    df, category, seq_length=10)

                logging.info(
                    f"Length of word_to_idx for category '{category}' : {len(word_to_idx)}")
                logging.info(f"Using vocab_size: {vocab_size} for training")

                if sequences:
                    logging.info(f"Training model for category: {category}")
                    train_model(sequences, vocab_size, model, batch_size=128,
                                num_epochs=10, model_dir=args.model_dir)

                    all_models_state[category] = model.state_dict()
                    logging.info(
                        f"Completed training for category: {category}")

        model_save_path = os.path.join(args.model_dir, 'all_models.tar')
        torch.save(all_models_state, model_save_path)
        logging.info(f"Model saved to {model_save_path}")

    except Exception as e:
        logging.error("Error in training: %s", e)
        raise e
