import os
import string
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Setup NLTK data directory explicitly for cloud environment
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

# Add custom nltk data path so nltk can find resources
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

@st.cache_data(show_spinner=False)
def download_nltk_resources():
    # Download punkt and stopwords only if missing
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path)

# Call the download function once, cached by Streamlit
download_nltk_resources()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    filtered_words = []
    for word in tokens:
        if word.isalnum():  # Keep only alphanumeric tokens
            if word not in stop_words and word not in string.punctuation:
                filtered_words.append(ps.stem(word))
    return " ".join(filtered_words)

def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["transformed_text"] = df["Text"].apply(transform_text)
    return df

def vectorize_text(text_series, save_path="artifacts/vectorizer.pkl"):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(text_series)

    # Save the vectorizer to artifacts folder
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)

    return X, vectorizer
