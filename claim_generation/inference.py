import torch
import torch.nn as nn
from collections import Counter
import os
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def build_vocab(data):
    try:
        counter = Counter()
        for text in data:
            counter.update(text.split())
        vocab = {word: i+1 for i, (word, _) in enumerate(counter.items())}
        vocab_size = len(vocab) + 1
        return vocab, vocab_size
    except Exception as e:
        print(f"Error in build_vocab: {e}")
        raise


def generate_text(model, vocab, initial_text, max_length=100):
    try:
        model.eval()
        words = initial_text.split()
        state_h, state_c = None, None

        device = next(model.parameters()).device
        vocab_inv = {idx: word for word, idx in vocab.items()}

        for _ in range(max_length):
            x = torch.tensor(
                [[vocab.get(word, 0) for word in words[-2:]]], dtype=torch.long).to(device)
            out, (state_h, state_c) = model(x, state_h, state_c)

            last_word_logits = out[0, -1]
            prob = torch.nn.functional.softmax(last_word_logits, dim=0).data
            word_id = torch.multinomial(prob, 1).item()
            word = vocab_inv.get(word_id, "<unk>")

            if word == '<end>':
                break

            words.append(word)

        return ' '.join(words)
    except Exception as e:
        logger.error(f"Error in generate_text: {e}")
        raise


def model_fn(model_dir):
    try:
        vocab_path = os.path.join(model_dir, 'vocab.json')
        with open(vocab_path, 'r') as vocab_file:
            vocab = json.load(vocab_file)
        vocab_size = len(vocab) + 1

        embed_size = 100
        hidden_size = 512
        num_layers = 2
        output_size = vocab_size

        model = LSTMModel(vocab_size, embed_size, hidden_size,
                          num_layers, output_size)
        with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
            model.load_state_dict(torch.load(f))

        return model, vocab
    except Exception as e:
        print(f"Error in model_fn: {e}")
        raise


def input_fn(request_body, request_content_type):
    try:
        if request_content_type == 'application/json':
            data = json.loads(request_body)
            initial_text = data['text']
            return initial_text
        raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        print(f"Error in input_fn: {e}")
        raise


def predict_fn(input_data, model_vocab_tuple):
    try:
        model, vocab = model_vocab_tuple

        logger.info(f"input_data: {input_data}")
        logger.info(f"model: {model}")

        generated_text = generate_text(model, vocab, input_data)
        return generated_text
    except Exception as e:
        logger.error(f"Error in predict_fn: {e}")
        raise


def output_fn(prediction_output, accept):
    try:
        if accept == 'application/json':
            return json.dumps({'generated_text': prediction_output}), accept
        raise ValueError(f"Unsupported accept type: {accept}")
    except Exception as e:
        print(f"Error in output_fn: {e}")
        raise
