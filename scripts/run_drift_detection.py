"""Implement PSI (Population Stability Index) to detect distribution drift."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import os

# Ensure imports work from the root folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from phishing_detector.features.build import extract_features

OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

def calculate_psi(expected, actual, buckets=10):
    """Calculate the Population Stability Index (PSI)."""
    def scale_range(input_arr, min_val, max_val):
        input_arr += -(np.min(input_arr))
        if np.max(input_arr) == 0:
            return np.zeros_like(input_arr) # Handle constant arrays safely
        input_arr /= np.max(input_arr) / (max_val - min_val)
        input_arr += min_val
        return input_arr

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Add small epsilon to avoid division by zero
    expected_percents = np.clip(expected_percents, 1e-4, None)
    actual_percents = np.clip(actual_percents, 1e-4, None)

    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi_value

def main():
    print("Loading datasets and models...")
    
    # 1. Load Historical Data (PhishTank)
    df_old = pd.read_csv("data/processed/temporal_benchmark.csv")
    
    # --- THE FIX: Parse timestamps and extract year ---
    df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], format='mixed', utc=True)
    df_old["year"] = df_old["timestamp"].dt.year
    # -------------------------------------------------
    
    # Filter to old data and sample
    df_old = df_old[df_old["year"] < 2024]
    if len(df_old) > 2000:
        df_old = df_old.sample(n=2000, random_state=42)
    
    # 2. Load New Data (PhiUSIIL)
    df_new = pd.read_csv("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
    if len(df_new) > 2000:
        df_new = df_new.sample(n=2000, random_state=42)
    url_col = next((c for c in df_new.columns if c.lower() == 'url'), None)
    
    print("Loading combined model...")
    model = joblib.load("models/combined_xgboost_phish_model.pkl")
    feature_cols = joblib.load("models/model_features.pkl")

    def get_probs(df, url_col_name):
        X_list = []
        for u in df[url_col_name]:
            feats = extract_features(str(u))
            feats.pop("tld", None)
            X_list.append(feats)
        X = pd.DataFrame(X_list)[feature_cols]
        return model.predict_proba(X)[:, 1]

    print("Calculating predictions for historical data...")
    probs_old = get_probs(df_old, "url")
    
    print("Calculating predictions for new data...")
    probs_new = get_probs(df_new, url_col)

    # 3. Calculate PSI
    psi = calculate_psi(probs_old, probs_new)
    
    print(f"\n=== DRIFT DETECTION ===")
    print(f"Population Stability Index (PSI): {psi:.4f}")
    
    if psi < 0.1:
        print("Status: No significant drift detected. Model is stable.")
    elif psi < 0.25:
        print("Status: Moderate drift detected. Calibration recommended.")
    else:
        print("Status: SEVERE DRIFT DETECTED! Retraining or Adaptation mandatory.")

    with open(OUT / "drift_detection.txt", "w") as f:
        f.write(f"PSI: {psi:.4f}\n")

if __name__ == "__main__":
    main()