import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)

    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)


def main():
    try:
        df = pd.read_csv('twitter_training.csv',
                         encoding='utf-8', on_bad_lines='skip', usecols=['Emotion', 'Quote'])
    except UnicodeDecodeError:
        try:
            df = pd.read_csv('twitter_training.csv',
                             encoding='latin1', on_bad_lines='skip', usecols=['Emotion', 'Quote'])
        except UnicodeDecodeError:
            df = pd.read_csv('twotter_training.csv',
                             encoding='ISO-8859-1', on_bad_lines='skip', usecols=['Emotion', 'Quote'])

    df.dropna(subset=['Quote'], inplace=True)

    df['Processed_Quote'] = df['Quote'].apply(preprocess_text)

    df.drop(columns=['Quote'], inplace=True)

    df.to_csv('twitter_training_processed.csv', index=False)

    print("Preprocessing complete. Processed file saved as 'twitter_training_processed.csv'.")


if __name__ == "__main__":
    main()
