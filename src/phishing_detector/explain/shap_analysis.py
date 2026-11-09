"""SHAP analysis wrapper for feature importance."""
import shap
import numpy as np

def get_shap_importance(model, X_data):
    """Calculate mean absolute SHAP values for a given dataset."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    
    # Calculate mean absolute SHAP value per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Return as a dictionary
    return dict(zip(X_data.columns, mean_shap))