import nltk
import os
import string
import re
import joblib

# Set a specific nltk_data download directory (helps in cloud environments like Streamlit)
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# Add this path to nltk data paths so nltk can find the downloaded files
nltk.data.path.append(nltk_data_path)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():  # Remove special characters
            if word not in stop_words and word not in string.punctuation:
                y.append(ps.stem(word))
    return " ".join(y)

def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["transformed_text"] = df["Text"].apply(transform_text)
    return df

def vectorize_text(text_series, save_path="artifacts/vectorizer.pkl"):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(text_series)

    # Save the vectorizer
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)

    return X, vectorizer
