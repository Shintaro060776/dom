import os
import joblib
import json
import traceback


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    vectorizer = joblib.load(os.path.join(
        model_dir, 'tfidf_vectorizer.joblib'))
    return model, vectorizer


def input_fn(request_body, request_content_type):
    if request_content_type == 'text/plain':
        return [request_body]
    else:
        raise ValueError(
            "Unsupported content type: {}".format(request_content_type))


def predict_fn(input_data, model_vectorizer):
    try:
        model, vectorizer = model_vectorizer
        X = vectorizer.transform(input_data)
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        traceback.print_exc()
        raise e


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction.tolist())
    return str(prediction)
