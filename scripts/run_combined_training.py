"""Domain Adaptation: Combine original data with PhiUSIIL to improve generalization."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from phishing_detector.calibration.metrics import expected_calibration_error
from phishing_detector.features.build import extract_features

warnings.filterwarnings("ignore")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    print("1. Loading original temporal dataset...")
    df_original = pd.read_csv("data/processed/temporal_benchmark.csv")
    # We only need url and label
    df_original = df_original[["url", "label"]]
    
    print("2. Loading PhiUSIIL dataset...")
    df_phi = pd.read_csv("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
    url_col = next((c for c in df_phi.columns if c.lower() == 'url'), None)
    df_phi = df_phi[[url_col, 'label']].rename(columns={url_col: 'url'})
    
    # Flip PhiUSIIL labels (0=Phish -> 1, 1=Legit -> 0)
    df_phi['label'] = df_phi['label'].apply(lambda x: 1 if x == 0 else 0)
    
    # To keep training fast, we sample 20,000 from PhiUSIIL
    df_phi_sample = df_phi.sample(n=20000, random_state=42)
    
    print("3. Combining datasets...")
    df_combined = pd.concat([df_original, df_phi_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   Combined dataset size: {len(df_combined)} URLs")

    print("4. Extracting features for combined dataset (This takes ~1 min)...")
    X_list = []
    for u in tqdm(df_combined["url"], desc="Extracting"):
        feats = extract_features(str(u))
        feats.pop("tld", None)
        X_list.append(feats)
    
    X = pd.DataFrame(X_list)
    y = df_combined["label"].values

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("5. Training XGBoost on Combined Data...")
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    print("6. Evaluating Raw Model on Combined Test Set...")
    proba_raw = model.predict_proba(X_test)[:, 1]
    pred_raw = (proba_raw >= 0.5).astype(int)
    
    acc_raw = accuracy_score(y_test, pred_raw)
    f1_raw = f1_score(y_test, pred_raw)
    ece_raw = expected_calibration_error(y_test, proba_raw, n_bins=15)
    
    print(f"   [Raw Combined Model] Accuracy: {acc_raw:.4f} | F1: {f1_raw:.4f} | ECE: {ece_raw:.4f}")

    print("7. Calibrating the Combined Model...")
    # Split test set for calibration and eval
    X_warmup, X_eval, y_warmup, y_eval = train_test_split(X_test, y_test, test_size=0.8, random_state=42, stratify=y_test)
    
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_warmup, y_warmup)
    
    proba_cal = calibrated_model.predict_proba(X_eval)[:, 1]
    pred_cal = (proba_cal >= 0.5).astype(int)
    
    acc_cal = accuracy_score(y_eval, pred_cal)
    f1_cal = f1_score(y_eval, pred_cal)
    ece_cal = expected_calibration_error(y_eval, proba_cal, n_bins=15)
    
    print(f"   [Calibrated Combined Model] Accuracy: {acc_cal:.4f} | F1: {f1_cal:.4f} | ECE: {ece_cal:.4f}")

    # Save the new robust model
    joblib.dump(model, "models/combined_xgboost_phish_model.pkl")
    print("\n✅ New combined model saved to models/combined_xgboost_phish_model.pkl")

if __name__ == "__main__":
    main()