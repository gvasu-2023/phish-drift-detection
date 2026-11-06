"""Ablation: How much warm-up data is needed to fix calibration?"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
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
    
    phish_years = df[df["label"] == 1]["year"]
    min_year, max_year = phish_years.min(), phish_years.max()
    np.random.seed(42)
    benign_mask = df["label"] == 0
    df.loc[benign_mask, "year"] = np.random.randint(min_year, max_year, size=benign_mask.sum())

    SPLIT_YEAR = 2024
    train_df = df[df["year"] < SPLIT_YEAR]
    test_df = df[df["year"] >= SPLIT_YEAR]
    
    # Balanced Training
    train_pos = train_df[train_df["label"] == 1].sample(n=5000, random_state=42)
    train_neg = train_df[train_df["label"] == 0].sample(n=5000, random_state=42)
    train_bal = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

    # Dynamically find how many we actually have, to avoid crashing
    max_pos = len(test_df[test_df["label"] == 1])
    max_neg = len(test_df[test_df["label"] == 0])
    
    n_pos = min(5000, max_pos)
    n_neg = min(45000, max_neg)
    
    # Imbalanced test set (roughly 10% phish)
    test_pos = test_df[test_df["label"] == 1].sample(n=n_pos, random_state=42)
    test_neg = test_df[test_df["label"] == 0].sample(n=n_neg, random_state=42) 
    test_imbal = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

    drop_cols = ["url", "timestamp", "year", "label", "tld"]
    X_train = train_bal.drop(columns=[c for c in drop_cols if c in train_bal.columns])
    y_train = train_bal["label"].values
    X_test = test_imbal.drop(columns=[c for c in drop_cols if c in test_imbal.columns])
    y_test = test_imbal["label"].values

    print("Training Base XGBoost...")
    base_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    base_model.fit(X_train, y_train)

    # Split off a fixed 20% evaluation set
    X_cal_pool, X_eval, y_cal_pool, y_eval = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
    )

    # Fractions of the TOTAL TEST SET to use for warm-up
    fractions = [0.10, 0.05, 0.01, 0.005, 0.001]
    results = []

    for frac in fractions:
        n_samples = int(frac * len(test_imbal))
        n_samples = min(n_samples, len(X_cal_pool)) # Cap at available pool
        
        # Sample warm-up data
        if n_samples < 10:
            print(f"Skipping fraction {frac} (too few samples)")
            continue
            
        X_warmup, _, y_warmup, _ = train_test_split(
            X_cal_pool, y_cal_pool, train_size=n_samples, random_state=42, stratify=y_cal_pool
        )
        
        # Fit calibrator
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_warmup, y_warmup)
        
        # Evaluate
        proba_cal = calibrated_model.predict_proba(X_eval)[:, 1]
        ece_cal = expected_calibration_error(y_eval, proba_cal, n_bins=15)
        
        results.append({
            "Warm-up %": f"{frac*100:.1f}%",
            "Warm-up Samples": n_samples,
            "ECE (Calibrated)": f"{ece_cal:.5f}"
        })

    res_df = pd.DataFrame(results)
    print("\n=== ABLATION: WARM-UP DATA EFFICIENCY ===")
    print(res_df.to_string(index=False))
    
    res_df.to_csv(OUT / "calibration_ablation.csv", index=False)
    print(f"\n✅ Table saved to {OUT / 'calibration_ablation.csv'}")

if __name__ == "__main__":
    main()