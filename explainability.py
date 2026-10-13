# EV Charging Demand Prediction - SHAP Explainability

import joblib
import shap
import pandas as pd


def load_explainer(model_path):
    """Load the trained classification model and create a SHAP explainer."""
    model = joblib.load(model_path)
    return shap.TreeExplainer(model)


def get_shap_values(explainer, X):
    """Calculate SHAP values for the supplied feature data."""
    return explainer.shap_values(X)
