
import joblib


def load_model(model_path):
    """Load the trained regression model."""
    return joblib.load(model_path)


def predict_demand(model, X):
    """Predict numerical EV charging demand."""
    return model.predict(X)
