"""Evaluation metrics for models."""
from sklearn.metrics import accuracy_score, f1_score
from phishing_detector.calibration.metrics import expected_calibration_error

def evaluate_predictions(y_true, y_proba):
    """Calculate standard evaluation metrics."""
    y_pred = (y_proba >= 0.5).astype(int)
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "ece": float(expected_calibration_error(y_true, y_proba, n_bins=15))
    }