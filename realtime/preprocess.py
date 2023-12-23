import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

try:
    df = pd.read_csv('data.csv')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('data.csv', encoding='latin1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv('data.csv', encoding='ISO-8859-1')
        except UnicodeDecodeError:
            df = pd.read_csv('data.csv', encoding='cp1252')


df['Tweet'] = df['Tweet'].astype(str)


def clean_tweet(tweet):

    if not tweet or pd.isna(tweet):
        return ""

    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)

    tokenizer = TweetTokenizer(
        preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    stemmer = PorterStemmer()
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stop_words and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return " ".join(tweets_clean)


df['clean_tweet'] = df['Tweet'].apply(clean_tweet)

df = df[['Sentiment', 'clean_tweet']]

df.to_csv('preprocessed.csv', index=False)
