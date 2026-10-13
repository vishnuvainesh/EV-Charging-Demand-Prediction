
import pandas as pd


def encode_location(df):
    """Convert Location categories into numerical columns."""
    return pd.get_dummies(df, columns=["Location"], dtype=int)
