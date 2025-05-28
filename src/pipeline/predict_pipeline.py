import joblib
import os
from src.components.data_transformation import transform_text
import numpy as np

MODEL_PATH = "artifacts/model.pkl"
VECTORIZER_PATH = "artifacts/vectorizer.pkl"

def load_model(path):
    return joblib.load(path)

def predict_message(text):
    # Load artifacts
    model = load_model(MODEL_PATH)
    vectorizer = load_model(VECTORIZER_PATH)

    # Clean and transform the text
    transformed_text = transform_text(text)
    vectorized_input = vectorizer.transform([transformed_text])

    # Predict
    prediction = model.predict(vectorized_input)[0]
    return "Spam" if prediction == 1 else "Ham"
