"""Cross-Dataset Generalization: Test the model on the PhiUSIIL benchmark (Local CSV)."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import sys
import os

# Ensure imports work from the root folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from phishing_detector.calibration.metrics import expected_calibration_error
from phishing_detector.features.build import extract_features

warnings.filterwarnings("ignore")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    print("1. Loading PhiUSIIL Dataset from local CSV...")
    csv_path = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find {csv_path}.")
        print("Please ensure the CSV is placed in the data/raw/ folder and named correctly.")
        return

    df = pd.read_csv(csv_path)
    print(f"   Successfully loaded {len(df)} URLs from PhiUSIIL.")
    
    # Find the URL column dynamically (usually 'URL')
    url_col = next((c for c in df.columns if c.lower() == 'url'), None)
    if not url_col:
        print(f"❌ Error: Could not find 'URL' column. Columns available: {df.columns}")
        return
        
    # IMPORTANT: In PhiUSIIL, 0 = Phishing, 1 = Legitimate. 
    # We must flip this to match our pipeline (1 = Phishing, 0 = Legitimate).
    df['label'] = df['label'].apply(lambda x: 1 if x == 0 else 0)
    
    # Sample 2,000 from each class to keep evaluation fast
    pos_samples = df[df['label'] == 1].sample(n=2000, random_state=42)
    neg_samples = df[df['label'] == 0].sample(n=2000, random_state=42)
    
    external_df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   Created balanced external test set of {len(external_df)} URLs (50% Phish).")

    print("\n2. Extracting features for external URLs...")
    X_ext_list = []
    for u in tqdm(external_df[url_col], desc="Extracting"):
        feats = extract_features(str(u))
        feats.pop("tld", None)
        X_ext_list.append(feats)
    
    X_ext = pd.DataFrame(X_ext_list)
    y_ext = external_df["label"].values

    print("\n3. Loading Robust Model...")
    model = joblib.load("models/robust_xgboost_phish_model.pkl")
    feature_cols = joblib.load("models/model_features.pkl")
    
    # Ensure the columns match exactly in the right order
    X_ext = X_ext[feature_cols]

    print("\n4. Evaluating Zero-Shot Performance (No Retraining)...")
    proba_raw = model.predict_proba(X_ext)[:, 1]
    pred_raw = (proba_raw >= 0.5).astype(int)
    
    acc_raw = accuracy_score(y_ext, pred_raw)
    f1_raw = f1_score(y_ext, pred_raw)
    ece_raw = expected_calibration_error(y_ext, proba_raw, n_bins=15)

    print(f"   [Raw Model] Accuracy: {acc_raw:.4f} | F1: {f1_raw:.4f} | ECE: {ece_raw:.4f}")

    print("\n5. Applying Lightweight Calibration to External Data...")
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
        "Dataset": "PhiUSIIL (UCI)",
        "Acc (Raw)": f"{acc_raw:.4f}",
        "F1 (Raw)": f"{f1_raw:.4f}",
        "ECE (Raw)": f"{ece_raw:.4f}",
        "Acc (Cal)": f"{acc_cal:.4f}",
        "F1 (Cal)": f"{f1_cal:.4f}",
        "ECE (Cal)": f"{ece_cal:.4f}"
    }]
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "cross_dataset_phiusiil.csv", index=False)
    print(f"\n✅ Results saved to {OUT / 'cross_dataset_phiusiil.csv'}")

if __name__ == "__main__":
    main()