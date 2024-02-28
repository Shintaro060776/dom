import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])

    args, _ = parser.parse_known_args()

    df = pd.read_csv(os.path.join(args.train, 'preprocessed_new_and_new.csv'))

    df.fillna('', inplace=True)

    X = df['Quote']
    y = df['Category']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    joblib.dump(vectorizer, os.path.join(
        args.model_dir, 'tfidf_vectorizer.joblib'))
