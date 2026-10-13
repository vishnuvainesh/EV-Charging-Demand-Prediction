
import pandas as pd


def load_data(file_path):
    """Load the cleaned EV charging dataset."""
    return pd.read_csv(file_path)


def prepare_features(df):
    """Prepare input features for the trained classification model."""
    X = df[
        [
            "UserID",
            "ChargerID",
            "ChargerCompany",
            "Location",
            "ChargerType",
            "Duration",
        ]
    ]

    X = pd.get_dummies(X, columns=["Location"], dtype=int)

    return X
