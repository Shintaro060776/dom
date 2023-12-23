import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import logging

logging.basicConfig(level=logging.INFO)

df = pd.read_csv('preprocessed.csv')
df = df.dropna(subset=['clean_tweet'])

X = df['clean_tweet']
y = df['Sentiment']

logging.info("データを分離しています...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', SVC()),
])

parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'clf__C': [1, 10, 100],
}

logging.info("グリッドサーチを開始します...")
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=3)

grid_search.fit(X_train, y_train)

logging.info("予測と評価を行います")
predictions = grid_search.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
