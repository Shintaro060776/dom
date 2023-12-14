import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv('processed_after_data.csv')
data['Text'].fillna('', inplace=True)

X = data['Text']
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

param_grid = {
    'tfidf__max_features': [3000, 5000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
predictions = grid_search.predict(X_test)
print(classification_report(y_test, predictions))
print(f'Accuracy: {accuracy_score(y_test, predictions)}')

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'emotion_analysis_model.pkl')
