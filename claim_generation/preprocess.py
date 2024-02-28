import pandas as pd
import re


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'@\w+\s', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


try:
    df = pd.read_csv('twcs.csv', encoding='utf-8', usecols=['text'])
except UnicodeDecodeError:
    try:
        df = pd.read_csv('twcs.csv', encoding='latin1', usecols=['text'])
    except UnicodeDecodeError:
        df = pd.read_csv('twcs.csv', encoding='ISO-8859-1', usecols=['text'])

df['text'] = df['text'].apply(preprocess_text)

df.to_csv('cleaned_dataset.csv', index=False)
