"""Compare multiple models (LR, RF, MLP, XGB) for Prior Shift Calibration."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from phishing_detector.calibration.metrics import expected_calibration_error

warnings.filterwarnings("ignore")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading temporal dataset...")
    df = pd.read_csv("data/processed/temporal_benchmark.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', utc=True)
    df["year"] = df["timestamp"].dt.year
    
    # Distribute Tranco dates
    phish_years = df[df["label"] == 1]["year"]
    min_year, max_year = phish_years.min(), phish_years.max()
    np.random.seed(42)
    benign_mask = df["label"] == 0
    df.loc[benign_mask, "year"] = np.random.randint(min_year, max_year, size=benign_mask.sum())

    SPLIT_YEAR = 2024
    train_df = df[df["year"] < SPLIT_YEAR]
    test_df = df[df["year"] >= SPLIT_YEAR]
    
    # Balanced Training (50/50)
    train_pos = train_df[train_df["label"] == 1].sample(n=5000, random_state=42)
    train_neg = train_df[train_df["label"] == 0].sample(n=5000, random_state=42)
    train_bal = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

    # Imbalanced Future Test (10% Phish, 90% Benign)
    test_pos = test_df[test_df["label"] == 1].sample(n=1000, random_state=42)
    test_neg = test_df[test_df["label"] == 0].sample(n=9000, random_state=42)
    test_imbal = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

    drop_cols = ["url", "timestamp", "year", "label", "tld"]
    X_train = train_bal.drop(columns=[c for c in drop_cols if c in train_bal.columns])
    y_train = train_bal["label"].values
    X_test = test_imbal.drop(columns=[c for c in drop_cols if c in test_imbal.columns])
    y_test = test_imbal["label"].values

    # Split test set into warm-up (calibration) and eval
    X_warmup, X_eval, y_warmup, y_eval = train_test_split(
        X_test, y_test, test_size=0.8, random_state=42, stratify=y_test
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    }

    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # 1. Uncalibrated Evaluation
        proba_raw = model.predict_proba(X_eval)[:, 1]
        ece_raw = expected_calibration_error(y_eval, proba_raw, n_bins=15)
        acc_raw = accuracy_score(y_eval, (proba_raw >= 0.5).astype(int))
        
        # 2. Calibrated Evaluation (Isotonic)
        print(f"  Applying Isotonic Calibration to {name}...")
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_warmup, y_warmup)
        
        proba_cal = calibrated_model.predict_proba(X_eval)[:, 1]
        ece_cal = expected_calibration_error(y_eval, proba_cal, n_bins=15)
        acc_cal = accuracy_score(y_eval, (proba_cal >= 0.5).astype(int))
        
        results.append({
            "Model": name,
            "Acc (Raw)": f"{acc_raw:.4f}",
            "ECE (Raw)": f"{ece_raw:.4f}",
            "Acc (Cal)": f"{acc_cal:.4f}",
            "ECE (Cal)": f"{ece_cal:.4f}"
        })

    # Display Results
    res_df = pd.DataFrame(results)
    print("\n=== MODEL COMPARISON: PRIOR SHIFT CALIBRATION ===")
    print(res_df.to_string(index=False))
    
    res_df.to_csv(OUT / "model_comparison.csv", index=False)
    print(f"\n✅ Table saved to {OUT / 'model_comparison.csv'}")

if __name__ == "__main__":
    main()