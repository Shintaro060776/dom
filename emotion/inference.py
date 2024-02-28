import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
    return model, vectorizer


def input_fn(request_body, request_content_type):
    if request_content_type == 'text/plain':
        data = np.array([request_body])
        return data
    else:
        pass


def predict_fn(input_data, model):
    model, vectorizer = model

    input_data_tfidf = vectorizer.transform(input_data)
    predictions = model.predict(input_data_tfidf)
    labels = ['negative' if pred == 0 else 'positive' for pred in predictions]
    return labels


def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps(prediction)
    elif content_type == "text/plain":
        return str(prediction)
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))
