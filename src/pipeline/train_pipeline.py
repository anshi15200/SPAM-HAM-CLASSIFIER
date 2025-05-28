from src.components.data_ingestion import ingest_data

df = ingest_data("data/spam.csv", "artifacts/cleaned_data.csv")
from src.components.data_transformation import transform_dataset, vectorize_text

df = transform_dataset(df)
X, vectorizer = vectorize_text(df["transformed_text"])
y = df["Target"]

from src.components.model_trainer import train_and_evaluate_model

model = train_and_evaluate_model(X, y)
