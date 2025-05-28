import os
import streamlit as st
import nltk
from src.pipeline.predict_pipeline import predict_message


nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# 🔽 Streamlit UI
st.title("📩 SMS Spam Classifier")
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
