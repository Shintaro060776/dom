import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

file_path = 'processed.csv'
data = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')

data.columns = ['Sentiment', 'Text']


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


data['Text'] = data['Text'].apply(preprocess_text)

processed_file_path = 'processed_after_data.csv'
data.to_csv(processed_file_path, index=False)

print("Processed data saved to:", processed_file_path)
