"""Multi-Year Temporal Decay: Proving how accuracy and calibration degrade year-by-year."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings

from phishing_detector.calibration.metrics import expected_calibration_error

warnings.filterwarnings("ignore")
FIG = Path("results/figures")
OUT = Path("results/tables")
FIG.mkdir(parents=True, exist_ok=True)
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

    # 1. Base Training Set: Everything BEFORE 2023
    SPLIT_YEAR = 2023
    train_df = df[df["year"] < SPLIT_YEAR]
    
    train_pos_pool = train_df[train_df["label"] == 1]
    train_neg_pool = train_df[train_df["label"] == 0]
    
    n_train_samples = min(5000, len(train_pos_pool), len(train_neg_pool))
    print(f"Available historical data before {SPLIT_YEAR}: {len(train_pos_pool)} Phish, {len(train_neg_pool)} Benign.")
    print(f"Using {n_train_samples} samples for training.")
    
    train_pos = train_pos_pool.sample(n=n_train_samples, random_state=42)
    train_neg = train_neg_pool.sample(n=n_train_samples, random_state=42)
    train_bal = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

    drop_cols = ["url", "timestamp", "year", "label", "tld"]
    X_train = train_bal.drop(columns=[c for c in drop_cols if c in train_bal.columns])
    y_train = train_bal["label"].values

    print(f"\nTraining Base XGBoost on data < {SPLIT_YEAR}...")
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # 2. Evaluate Year-by-Year
    test_years = [2023, 2024, 2025, 2026]
    results = []

    for year in test_years:
        test_df = df[df["year"] == year]
        
        max_pos = len(test_df[test_df["label"] == 1])
        max_neg = len(test_df[test_df["label"] == 0])
        n = min(1000, max_pos, max_neg)
        
        if n < 50:
            continue

        test_pos = test_df[test_df["label"] == 1].sample(n=n, random_state=42)
        test_neg = test_df[test_df["label"] == 0].sample(n=n, random_state=42)
        test_bal = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

        X_test = test_bal.drop(columns=[c for c in drop_cols if c in test_bal.columns])
        y_test = test_bal["label"].values

        # Raw Evaluation
        proba_raw = model.predict_proba(X_test)[:, 1]
        acc_raw = accuracy_score(y_test, (proba_raw >= 0.5).astype(int))
        ece_raw = expected_calibration_error(y_test, proba_raw, n_bins=15)

        # Apply Lightweight Calibration (Changed to 50/50 split to prevent overfitting on small years)
        X_warmup, X_eval, y_warmup, y_eval = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
        
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_warmup, y_warmup)
        
        proba_cal = calibrated_model.predict_proba(X_eval)[:, 1]
        acc_cal = accuracy_score(y_eval, (proba_cal >= 0.5).astype(int))
        ece_cal = expected_calibration_error(y_eval, proba_cal, n_bins=15)

        results.append({
            "Year": year,
            "Acc (Raw)": acc_raw,
            "ECE (Raw)": ece_raw,
            "Acc (Calibrated)": acc_cal,
            "ECE (Calibrated)": ece_cal
        })

    res_df = pd.DataFrame(results)
    print("\n=== MULTI-YEAR TEMPORAL DECAY ===")
    print(res_df.to_string(index=False))
    res_df.to_csv(OUT / "multi_year_decay.csv", index=False)

    # 3. Plot the Decay Curve
    print("\nGenerating Multi-Year Decay Plot...")
    plt.figure(figsize=(8, 5))
    plt.plot(res_df["Year"], res_df["ECE (Raw)"], marker='o', color='red', linewidth=2, label='Raw Model (Uncalibrated)')
    plt.plot(res_df["Year"], res_df["ECE (Calibrated)"], marker='s', color='green', linewidth=2, label='Lightweight Calibrated')
    
    plt.xlabel("Deployment Year", fontsize=12)
    plt.ylabel("Expected Calibration Error (ECE)", fontsize=12)
    plt.title(f"Calibration Decay Over Time (Trained on <{SPLIT_YEAR})", fontsize=14)
    plt.xticks(test_years)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = FIG / "multi_year_decay.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Figure saved to {plot_path}")

if __name__ == "__main__":
    main()