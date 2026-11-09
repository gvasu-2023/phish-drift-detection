"""Cross-Dataset Generalization: Test the model on live OpenPhish data."""
import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

from phishing_detector.calibration.metrics import expected_calibration_error
from phishing_detector.features.build import extract_features

warnings.filterwarnings("ignore")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    print("1. Downloading live OpenPhish feed...")
    url = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        phish_urls = [line.strip() for line in response.text.split('\n') if line.strip()]
        print(f"   Downloaded {len(phish_urls)} live phishing URLs from OpenPhish.")
    except Exception as e:
        print(f"   Failed to download OpenPhish feed: {e}")
        return

    # To keep evaluation fast, we'll sample 2,000 phishing URLs
    np.random.seed(42)
    if len(phish_urls) > 2000:
        phish_urls = np.random.choice(phish_urls, size=2000, replace=False).tolist()

    print("2. Loading benign URLs from our local temporal dataset...")
    df_local = pd.read_csv("data/processed/temporal_benchmark.csv")
    benign_urls = df_local[df_local["label"] == 0]["url"].sample(n=len(phish_urls), random_state=42).tolist()

    # Combine into a balanced external test set
    external_df = pd.DataFrame({
        "url": phish_urls + benign_urls,
        "label": [1] * len(phish_urls) + [0] * len(benign_urls)
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"   Created balanced external test set of {len(external_df)} URLs.")

    print("3. Extracting features for external URLs...")
    X_ext_list = []
    for u in tqdm(external_df["url"], desc="Extracting"):
        feats = extract_features(u)
        feats.pop("tld", None)
        X_ext_list.append(feats)
    
    X_ext = pd.DataFrame(X_ext_list)
    y_ext = external_df["label"].values

    print("4. Loading Robust Model...")
    # We use the robust model trained with adversarial data
    model = joblib.load("models/robust_xgboost_phish_model.pkl")
    feature_cols = joblib.load("models/model_features.pkl")
    X_ext = X_ext[feature_cols]

    print("5. Evaluating Zero-Shot Performance (No Retraining)...")
    proba_raw = model.predict_proba(X_ext)[:, 1]
    pred_raw = (proba_raw >= 0.5).astype(int)
    
    acc_raw = accuracy_score(y_ext, pred_raw)
    f1_raw = f1_score(y_ext, pred_raw)
    ece_raw = expected_calibration_error(y_ext, proba_raw, n_bins=15)

    print(f"   [Raw Model] Accuracy: {acc_raw:.4f} | F1: {f1_raw:.4f} | ECE: {ece_raw:.4f}")

    print("6. Applying Lightweight Calibration to External Data...")
    # Split external data into warm-up and eval
    X_warmup, X_eval, y_warmup, y_eval = train_test_split(
        X_ext, y_ext, test_size=0.8, random_state=42, stratify=y_ext
    )
    
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_warmup, y_warmup)
    
    proba_cal = calibrated_model.predict_proba(X_eval)[:, 1]
    pred_cal = (proba_cal >= 0.5).astype(int)
    
    acc_cal = accuracy_score(y_eval, pred_cal)
    f1_cal = f1_score(y_eval, pred_cal)
    ece_cal = expected_calibration_error(y_eval, proba_cal, n_bins=15)

    print(f"   [Calibrated Model] Accuracy: {acc_cal:.4f} | F1: {f1_cal:.4f} | ECE: {ece_cal:.4f}")

    # Save results
    results = [{
        "Dataset": "OpenPhish (Live 2026)",
        "Acc (Raw)": f"{acc_raw:.4f}",
        "F1 (Raw)": f"{f1_raw:.4f}",
        "ECE (Raw)": f"{ece_raw:.4f}",
        "Acc (Cal)": f"{acc_cal:.4f}",
        "F1 (Cal)": f"{f1_cal:.4f}",
        "ECE (Cal)": f"{ece_cal:.4f}"
    }]
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "cross_dataset_openphish.csv", index=False)
    print(f"\n✅ Results saved to {OUT / 'cross_dataset_openphish.csv'}")

if __name__ == "__main__":
    main()