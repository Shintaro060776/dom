import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model


def input_fn(request_body, request_content_type):
    vectorizer = TfidfVectorizer()
    return vectorizer.transform([request_body])


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, content_type):
    return str(prediction[0])
