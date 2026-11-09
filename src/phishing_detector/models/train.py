"""Model training module."""
from xgboost import XGBClassifier

def get_xgboost_model(n_estimators=300, max_depth=6):
    """Returns a configured XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=0.1, 
        n_jobs=-1, 
        random_state=42
    )