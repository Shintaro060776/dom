import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import boto3
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_df', type=float, default=0.5)
    parser.add_argument('--C', type=float, default=1.0)

    return parser.parse_known_args()


if __name__ == '__main__':
    args, _ = parse_args()

    input_data_path = os.environ['SM_CHANNEL_TRAIN']
    output_model_path = os.environ['SM_MODEL_DIR']

    s3 = boto3.client('s3')
    bucket = 'realtime20090317'
    file_key = 'preprocessed.csv'
    s3.download_file(bucket, file_key, 'preprocessed.csv')

    df = pd.read_csv('preprocessed.csv')
    df = df.dropna(subset=['clean_tweet'])

    X = df['clean_tweet']
    y = df['Sentiment']

    logging.info("データを分離しています...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=args.max_df)),
        ('clf', SVC(C=args.C)),
    ])

    pipeline.fit(X_train, y_train)

    parameters = {
        'tfidf__max_df': (0.5, 0.75, 1.0),
        'clf__C': [1, 10, 100],
    }

    logging.info("グリッドサーチを開始します...")
    grid_search = GridSearchCV(
        pipeline, parameters, cv=5, n_jobs=-1, verbose=3)

    grid_search.fit(X_train, y_train)

    logging.info("予測と評価を行います")
    predictions = grid_search.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    joblib.dump(grid_search.best_estimator_, os.path.join(
        output_model_path, 'model.joblib'))
