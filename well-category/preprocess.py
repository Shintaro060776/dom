import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')

df = pd.read_csv('short.csv', usecols=['Category', 'Quote'])

df['Category'] = df['Category'].fillna('').apply(
    lambda x: x.split(',')[0].lower().strip())


def clean_text(text):
    text = re.sub(r'[^\w\s\'-]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()


df['Quote'] = df['Quote'].apply(clean_text)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def lemmatize_text(text):
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])


df['Quote'] = df['Quote'].apply(lemmatize_text)

df.to_csv('preprocessed_new.csv', index=False)
