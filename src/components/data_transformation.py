import os
import string
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('punkt', quiet=True, force=True)
nltk.download('stopwords', quiet=True, force=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_words = []
    for word in tokens:
        if word.isalnum():
            if word not in stop_words and word not in string.punctuation:
                filtered_words.append(ps.stem(word))
    return " ".join(filtered_words)

def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["transformed_text"] = df["Text"].apply(transform_text)
    return df

def vectorize_text(text_series, save_path="artifacts/vectorizer.pkl"):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(text_series)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)
    return X, vectorizer
