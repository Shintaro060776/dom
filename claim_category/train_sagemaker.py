import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_data(filename):
    df = pd.read_csv(filename)
    df['Processed_Quote'] = df['Processed_Quote'].fillna('')
    return df['Processed_Quote'], df['Emotion']


def main(args):
    data_dir = args.data_dir
    output_dir = args.model_dir

    X, y = load_data(os.path.join(data_dir, 'twitter_training_processed.csv'))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(
        f"Classification Report:\n{classification_report(y_test, predictions)}")

    # joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    joblib.dump((vectorizer, model), os.path.join(output_dir, "model.joblib"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    main(args)
