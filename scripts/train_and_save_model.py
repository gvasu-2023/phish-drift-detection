"""Train the final XGBoost model on all available data and save it for the app."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading data...")
    df = pd.read_csv("data/processed/temporal_benchmark.csv")
    
    # Drop non-feature columns
    drop_cols = ["url", "timestamp", "label", "tld"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"].values

    print("Training final XGBoost model...")
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    model.fit(X, y)

    # Save model and expected features list
    joblib.dump(model, MODEL_DIR / "xgboost_phish_model.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "model_features.pkl")
    
    print(f"✅ Model saved to {MODEL_DIR / 'xgboost_phish_model.pkl'}")

if __name__ == "__main__":
    main()