"""Demonstrate Prior Shift miscalibration and fix it with lightweight Isotonic Regression."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings

from phishing_detector.calibration.metrics import (
    expected_calibration_error, brier, reliability_bins,
)

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
    
    phish_years = df[df["label"] == 1]["year"]
    min_year, max_year = phish_years.min(), phish_years.max()
    np.random.seed(42)
    benign_mask = df["label"] == 0
    df.loc[benign_mask, "year"] = np.random.randint(min_year, max_year, size=benign_mask.sum())

    SPLIT_YEAR = 2024
    train_df = df[df["year"] < SPLIT_YEAR]
    test_df = df[df["year"] >= SPLIT_YEAR]
    
    # 1. Balanced Training Set
    train_pos = train_df[train_df["label"] == 1].sample(n=5000, random_state=42)
    train_neg = train_df[train_df["label"] == 0].sample(n=5000, random_state=42)
    train_bal = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

    # 2. Imbalanced Future Test Set (90% benign, 10% phish)
    print("Creating Real-World Imbalanced Test Set (10% Phish, 90% Benign)...")
    test_pos = test_df[test_df["label"] == 1].sample(n=1000, random_state=42)
    test_neg = test_df[test_df["label"] == 0].sample(n=9000, random_state=42)
    test_imbal = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

    drop_cols = ["url", "timestamp", "year", "label", "tld"]
    X_train = train_bal.drop(columns=[c for c in drop_cols if c in train_bal.columns])
    y_train = train_bal["label"].values
    
    X_test = test_imbal.drop(columns=[c for c in drop_cols if c in test_imbal.columns])
    y_test = test_imbal["label"].values

    print("Training XGBoost on Balanced Past Data...")
    base_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    base_model.fit(X_train, y_train)

    print("\nApplying Lightweight Isotonic Calibration on a small warm-up set...")
    X_warmup, X_eval, y_warmup, y_eval = train_test_split(
        X_test, y_test, test_size=0.8, random_state=42, stratify=y_test
    )
    
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_warmup, y_warmup)
    
    proba_base_eval = base_model.predict_proba(X_eval)[:, 1]
    proba_cal = calibrated_model.predict_proba(X_eval)[:, 1]
    
    ece_base_eval = expected_calibration_error(y_eval, proba_base_eval, n_bins=10)
    ece_cal = expected_calibration_error(y_eval, proba_cal, n_bins=10)

    print(f"[Before Adaptation] ECE: {ece_base_eval:.4f}")
    print(f"[After Adaptation]  ECE: {ece_cal:.4f}")
    
    # 6. Plot Before vs After Reliability Diagram (Fixed to remove sparse bins)
    print("\nGenerating Before/After Reliability Diagram...")
    bins_base = reliability_bins(y_eval, proba_base_eval, n_bins=10)
    bins_cal = reliability_bins(y_eval, proba_cal, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # Plot baseline (unadapted) - filter out empty bins
    for i in range(len(bins_base['mean_confidence'])):
        if bins_base['count'][i] > 5: # Only plot bins with > 5 samples
            plt.scatter(bins_base['mean_confidence'][i], bins_base['mean_accuracy'][i], marker='s', color='red', s=60)
    plt.plot([bins_base['mean_confidence'][i] for i in range(len(bins_base['mean_confidence'])) if bins_base['count'][i] > 5], 
             [bins_base['mean_accuracy'][i] for i in range(len(bins_base['mean_accuracy'])) if bins_base['count'][i] > 5], 
             color='red', linewidth=2, label=f"Unadapted (ECE = {ece_base_eval:.4f})")
             
    # Plot calibrated - filter out empty bins
    for i in range(len(bins_cal['mean_confidence'])):
        if bins_cal['count'][i] > 5:
            plt.scatter(bins_cal['mean_confidence'][i], bins_cal['mean_accuracy'][i], marker='o', color='green', s=60)
    plt.plot([bins_cal['mean_confidence'][i] for i in range(len(bins_cal['mean_confidence'])) if bins_cal['count'][i] > 5], 
             [bins_cal['mean_accuracy'][i] for i in range(len(bins_cal['mean_accuracy'])) if bins_cal['count'][i] > 5], 
             color='green', linewidth=2, label=f"Lightweight Adapt (ECE = {ece_cal:.4f})")
             
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (Actual)", fontsize=12)
    plt.title("Reliability Diagram: Prior Shift Adaptation", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = FIG / "prior_shift_reliability.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Figure saved to {plot_path}")

if __name__ == "__main__":
    main()