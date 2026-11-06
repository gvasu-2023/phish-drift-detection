"""Temporal Evaluation: Train on past, test on future with balanced priors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings

from phishing_detector.calibration.metrics import (
    expected_calibration_error, brier, reliability_bins,
)

warnings.filterwarnings("ignore")
OUT = Path("results/tables")
FIG = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading temporal dataset...")
    df = pd.read_csv("data/processed/temporal_benchmark.csv")
    
    # 1. Convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', utc=True)
    df["year"] = df["timestamp"].dt.year
    
    # 2. Fix Tranco dates (uniformly distribute)
    phish_years = df[df["label"] == 1]["year"]
    min_year, max_year = phish_years.min(), phish_years.max()
    np.random.seed(42)
    benign_mask = df["label"] == 0
    df.loc[benign_mask, "year"] = np.random.randint(min_year, max_year, size=benign_mask.sum())

    # 3. Temporal Split (Past vs Future)
    SPLIT_YEAR = 2024
    train_df = df[df["year"] < SPLIT_YEAR]
    test_df = df[df["year"] >= SPLIT_YEAR]
    
    # 4. BALANCE THE DATA
    print("\nBalancing datasets to isolate Concept Drift from Prior Shift...")
    
    def get_balanced_set(subset, max_n=5000):
        pos = subset[subset["label"] == 1]
        neg = subset[subset["label"] == 0]
        # Take the minimum of max_n, available positives, or available negatives
        n = min(max_n, len(pos), len(neg))
        return pd.concat([pos.sample(n=n, random_state=42), 
                          neg.sample(n=n, random_state=42)]).sample(frac=1, random_state=42)
        
    train_bal = get_balanced_set(train_df, max_n=5000)
    test_bal = get_balanced_set(test_df, max_n=5000)

    print(f"Train set (Past, < {SPLIT_YEAR}): {len(train_bal):,} rows (50% Phish)")
    print(f"Test set  (Future, >= {SPLIT_YEAR}): {len(test_bal):,} rows (50% Phish)")

    # 5. Prepare features
    drop_cols = ["url", "timestamp", "year", "label", "tld"]
    X_train = train_bal.drop(columns=[c for c in drop_cols if c in train_bal.columns])
    y_train = train_bal["label"].values
    
    X_test = test_bal.drop(columns=[c for c in drop_cols if c in test_bal.columns])
    y_test = test_bal["label"].values
    
    # 6. Train Model
    print("\nTraining XGBoost on Past data...")
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    # 7. Evaluate on Future data
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "split_year": SPLIT_YEAR,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "ece": float(expected_calibration_error(y_test, proba, n_bins=15)),
        "brier": brier(y_test, proba),
    }
    
    print("\n=== BALANCED TEMPORAL EVALUATION RESULTS ===")
    print(json.dumps(metrics, indent=2))
    
    # Save results
    (OUT / f"temporal_balanced_{SPLIT_YEAR}.json").write_text(json.dumps(metrics, indent=2))
    
    # 8. PLOT RELIABILITY DIAGRAM
    print("\nGenerating Reliability Diagram...")
    bins = reliability_bins(y_test, proba, n_bins=10)
    
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.plot(bins['mean_confidence'], bins['mean_accuracy'], 
             marker='o', color='red', linewidth=2, label=f"XGBoost (ECE = {metrics['ece']:.4f})")
             
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (Actual)", fontsize=12)
    plt.title(f"Reliability Diagram: Train <{SPLIT_YEAR}, Test >={SPLIT_YEAR}", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = FIG / "temporal_reliability_diagram.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Figure saved to {plot_path}")

if __name__ == "__main__":
    main()