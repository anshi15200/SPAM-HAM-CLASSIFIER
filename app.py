import os
import streamlit as st
from src.pipeline.predict_pipeline import predict_message
import nltk
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up a persistent nltk data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Add nltk data directory to nltk paths if not already present
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Download punkt and stopwords only if missing, and download them to nltk_data_dir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

st.set_page_config(page_title="Spam Ham Classifier", layout="centered")

st.title("📩 SMS Spam Classifier")
st.markdown("Detect whether a given message is **Spam** or **Ham (Not Spam)** using a trained ML model.")

# Input box
user_input = st.text_area("Enter your message below:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict_message(user_input)
        if result == "Spam":
            st.error("🚫 It's SPAM!")
        else:
            st.success("✅ It's HAM (not spam).")
