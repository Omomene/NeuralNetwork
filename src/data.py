import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(path):
    
    return pd.read_csv(path)


def clean_dataset(df):

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    numeric_cols = ["price", "height", "width", "depth"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates()

    return df


def filter_categories(df, categories):
    
    return df[df["category"].isin(categories)].copy()


def prepare_ml_data(df):

    X = df[
        [
            "price",
            "height",
            "width",
            "depth"
        ]
    ]

    encoder = LabelEncoder()

    y = encoder.fit_transform(df["category"])

    return X, y, encoder


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )