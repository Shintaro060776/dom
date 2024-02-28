import joblib
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def model_fn(model_dir):
    try:
        vectorizer, model = joblib.load(
            os.path.join(model_dir, "model.joblib"))
        return vectorizer, model
    except Exception as e:
        raise RuntimeError("Failed to load model: {}".format(e))


def predict_fn(input_data, model):
    vectorizer, clf_model = model
    try:
        # vectorizer = TfidfVectorizer(max_features=5000)
        input_data_transformed = vectorizer.transform(input_data)
        predictions = clf_model.predict(input_data_transformed)
        return predictions
    except Exception as e:
        raise RuntimeError("Failed to predict: {}".format(e))


def input_fn(request_body, request_content_type):
    if request_content_type == "text/plain":
        return np.array([request_body])
    else:
        raise ValueError("This model only supports text/plain input")


def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction.tolist()), accept
    raise ValueError("This model only supports JSON output")
