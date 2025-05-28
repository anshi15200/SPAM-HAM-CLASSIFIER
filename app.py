import streamlit as st
from src.pipeline.predict_pipeline import predict_message

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
