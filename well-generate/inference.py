import os
import torch
import torch.nn as nn
from collections import Counter
from nltk.tokenize import word_tokenize
import json
import logging

word_to_idx = {}
idx_to_word = {}


def generate_text(input_text, model, word_to_idx, idx_to_word, max_length=50):
    model.eval()
    with torch.no_grad():
        tokenized_input = [word_to_idx.get(
            word, 1) for word in word_tokenize(input_text.lower())]
        generated_text = tokenized_input[:]

        for _ in range(max_length):
            input_tensor = torch.tensor(
                generated_text[-10:], dtype=torch.long).unsqueeze(0)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=-1)
            _, next_token_idx = torch.max(probabilities, dim=2)
            next_word_idx = next_token_idx[0, -1].item()

            if next_word_idx == word_to_idx.get("<end>", -1):
                break

            generated_text.append(next_word_idx)

        generated_words = [idx_to_word.get(
            idx, "<unk>") for idx in generated_text]
        return ' '.join(generated_words)


def load_idx_to_word(model_dir):
    global idx_to_word
    with open(os.path.join(model_dir, VOCAB_FILE)) as vocab_file:
        word_to_idx = json.load(vocab_file)
    idx_to_word = {i: word for word, i in word_to_idx.items()}


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
        return prediction


VOCAB_FILE = "vocab.json"
META_FILE = "meta.json"


def model_fn(model_dir):
    global word_to_idx

    try:
        import nltk
        nltk.download('punkt')
        with open(os.path.join(model_dir, META_FILE)) as meta_file:
            meta_data = json.load(meta_file)

        model = LSTMModel(
            vocab_size=meta_data['vocab_size'],
            embedding_dim=meta_data['embedding_dim'],
            hidden_dim=meta_data['hidden_dim']
        )

        with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
            model.load_state_dict(torch.load(f))

        with open(os.path.join(model_dir, VOCAB_FILE)) as vocab_file:
            word_to_idx = json.load(vocab_file)

        load_idx_to_word(model_dir)
        return model
    except Exception as e:
        logging.error(f"Error in model_fn: {e}")
        raise


def predict_fn(input_text, model):
    try:
        generated_text = generate_text(
            input_text, model, word_to_idx, idx_to_word)
        return generated_text
    except Exception as e:
        logging.error(f"Error in predict_fn: {e}")
        raise


def input_fn(request_body, request_content_type):
    try:
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)
            input_text = input_data['text']
            return input_text
        else:
            raise ValueError(
                'Unsupported content type: {}'.format(request_content_type))
    except Exception as e:
        logging.error(f"Error in input_fn: {e}")
        raise


def output_fn(prediction_output, accept):
    try:
        if accept == 'application/json':
            response = {'generated_text': prediction_output}
            return json.dumps(response), accept
        else:
            raise ValueError('Unsupported accept type: {}'.format(accept))
    except Exception as e:
        logging.error(f"Error in output_fn: {e}")
        raise
