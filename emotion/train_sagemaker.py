import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'))

    args = parser.parse_args()

    data = pd.read_csv(os.path.join(args.train, 'processed_after_data.csv'))
    data['Text'].fillna('', inplace=True)

    X = data['Text']
    y = data['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    X_test_tfidf = vectorizer.transform(X_test)
    predictions = model.predict(X_test_tfidf)
    print(classification_report(y_test, predictions))
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')

    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    joblib.dump(vectorizer, os.path.join(args.model_dir, 'vectorizer.joblib'))
