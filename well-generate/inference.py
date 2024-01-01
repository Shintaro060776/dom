import os
import torch
import torch.nn as nn
from collections import Counter
from nltk.tokenize import word_tokenize
import json


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
        model.word_to_idx = json.load(vocab_file)

    return model


def predict_fn(input_data, model):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.long)
        output = model(input_tensor)
        _, predicted_index = torch.max(output, 1)
        return predicted_index


def input_fn(request_body, request_content_type, model):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        tokenized_text = word_tokenize(input_data['text'].lower())

        input_data = [model.word_to_idx.get(
            word, 1) for word in tokenized_text]
        return input_data
    else:
        raise ValueError(
            'Unsupported content type: {}'.format(request_content_type))


def output_fn(prediction_output, accept):
    if accept == 'application/json':
        response = {'predicted_token_index': prediction_output.item()}
        return json.dumps(response), accept
    else:
        raise ValueError('Unsupported accept type: {}'.format(accept))
