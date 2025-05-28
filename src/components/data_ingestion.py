import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def ingest_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding='ISO-8859-1')

    # Drop irrelevant columns
    df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)

    # Rename columns
    df.rename(columns={"v1": "Target", "v2": "Text"}, inplace=True)

    # Encode Target column
    encoder = LabelEncoder()
    df["Target"] = encoder.fit_transform(df["Target"])

    # Drop duplicates
    df.drop_duplicates(keep="first", inplace=True)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df
