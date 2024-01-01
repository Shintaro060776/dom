import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    vocabulary = joblib.load(os.path.join(model_dir, 'tfidf_vocabulary.pkl'))
    return model, vocabulary


def input_fn(request_body, request_content_type):
    if request_content_type == 'text/plain':
        df = pd.DataFrame([request_body], columns=['text'])
        return df
    else:
        pass


def predict_fn(input_data, model_vocabulary):
    model, vocabulary = model_vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    X = vectorizer.transform(input_data['text'])

    predictions = model.predict(X)
    return predictions


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    return str(prediction)
