import os
import joblib
import logging
import json

logging.basicConfig(level=logging.INFO)


def model_fn(model_dir):
    try:
        model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        return model
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        raise


def input_fn(request_body, request_content_type):
    try:
        return request_body
    except Exception as e:
        logging.error(f"Input processing failed: {str(e)}")
        raise


def predict_fn(input_data, model):
    try:
        return model.predict([input_data])
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise


def output_fn(prediction, content_type):
    try:
        return json.dumps({"sentiment": str(prediction[0])})
    except Exception as e:
        logging.error(f"Output processing failed: {str(e)}")
        raise
