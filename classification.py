
import joblib


def load_model(model_path):
    """Load the trained Random Forest classification model."""
    return joblib.load(model_path)


def predict_demand(model, X):
    """Predict demand class: 0 = Normal, 1 = High."""
    return model.predict(X)


def predict_demand_probability(model, X):
    """Return probabilities for Normal and High Demand."""
    return model.predict_proba(X)
