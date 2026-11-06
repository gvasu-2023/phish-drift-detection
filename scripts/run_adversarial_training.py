"""Robust Tier 3: Adversarial Training to fix the 50% collapse."""
import pandas as pd
import numpy as np
import random
import string
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings

from phishing_detector.calibration.metrics import expected_calibration_error
from phishing_detector.features.build import extract_features

warnings.filterwarnings("ignore")
FIG = Path("results/figures")
OUT = Path("results/tables")
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

def perturb_url(url, attack_type):
    try:
        if not url.startswith("http"): url = "http://" + url
        parts = url.split("://")
        scheme, rest = parts[0], parts[1]
        
        if attack_type == "typosquatting":
            return f"{scheme}://{rest.replace('o', '0').replace('l', '1').replace('e', '3')}"
        elif attack_type == "subdomain_padding":
            rand_sub = ''.join(random.choices(string.ascii_lowercase, k=5))
            return f"{scheme}://{rand_sub}.secure.{rest}"
        elif attack_type == "path_obfuscation":
            rand_path = ''.join(random.choices(string.ascii_letters + string.digits, k=30))
            return f"{scheme}://{rest}/{rand_path}/login.php"
        return url
    except:
        return url

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
    
    # 1. Base Clean Training Data
    train_pos = train_df[train_df["label"] == 1].sample(n=5000, random_state=42)
    train_neg = train_df[train_df["label"] == 0].sample(n=5000, random_state=42)
    train_clean = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 2. Generate Adversarial Training Data
    print("Generating Adversarial Training Examples (10,000)...")
    attacks = ["typosquatting", "subdomain_padding", "path_obfuscation"]
    adv_data = []
    for _, row in train_clean.iterrows():
        attack = random.choice(attacks)
        adv_url = perturb_url(row["url"], attack)
        adv_data.append({"url": adv_url, "label": row["label"]})
    
    train_adv = pd.DataFrame(adv_data)
    
    # Combine Clean + Adversarial
    train_combined = pd.concat([train_clean[["url", "label"]], train_adv]).sample(frac=1, random_state=42)

    print(f"Extracting features for {len(train_combined)} training URLs...")
    X_train_list = []
    for url in tqdm(train_combined["url"], desc="Train Features"):
        feats = extract_features(url)
        feats.pop("tld", None)
        X_train_list.append(feats)
    
    X_train = pd.DataFrame(X_train_list)
    y_train = train_combined["label"].values

    # 3. Prepare Test Data (Same as Tier 3 for fair comparison)
    test_pos = test_df[test_df["label"] == 1].sample(n=1000, random_state=42)
    test_neg = test_df[test_df["label"] == 0].sample(n=1000, random_state=42)
    test_clean = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    train_cal, test_eval = train_test_split(test_clean, test_size=0.8, random_state=42, stratify=test_clean["label"])

    # 4. Train Robust Model
    print("\nTraining Robust XGBoost...")
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # 5. Fit Calibrator on Clean Warm-up
    X_cal_list = []
    for url in train_cal["url"]:
        feats = extract_features(url)
        feats.pop("tld", None)
        X_cal_list.append(feats)
    X_cal = pd.DataFrame(X_cal_list)[model.get_booster().feature_names]
    y_cal = train_cal["label"].values

    print("Calibrating Model...")
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_cal, y_cal)

    # 6. Evaluate on Adversarial Test Sets
    eval_attacks = ["clean", "typosquatting", "subdomain_padding", "path_obfuscation"]
    results = []

    for attack in eval_attacks:
        print(f"Evaluating: {attack.upper()}")
        if attack == "clean":
            eval_urls = test_eval["url"].tolist()
        else:
            eval_urls = [perturb_url(u, attack) for u in test_eval["url"]]
            
        X_eval_list = []
        for url in eval_urls:
            feats = extract_features(url)
            feats.pop("tld", None)
            X_eval_list.append(feats)
            
        X_eval = pd.DataFrame(X_eval_list)[model.get_booster().feature_names]
        y_eval = test_eval["label"].values
        
        proba_raw = model.predict_proba(X_eval)[:, 1]
        acc_raw = accuracy_score(y_eval, (proba_raw >= 0.5).astype(int))
        ece_raw = expected_calibration_error(y_eval, proba_raw, n_bins=15)
        
        proba_cal = calibrated_model.predict_proba(X_eval)[:, 1]
        acc_cal = accuracy_score(y_eval, (proba_cal >= 0.5).astype(int))
        ece_cal = expected_calibration_error(y_eval, proba_cal, n_bins=15)
        
        results.append({
            "Attack Type": attack,
            "Acc (Raw)": f"{acc_raw:.4f}",
            "ECE (Raw)": f"{ece_raw:.4f}",
            "Acc (Cal)": f"{acc_cal:.4f}",
            "ECE (Cal)": f"{ece_cal:.4f}"
        })

    res_df = pd.DataFrame(results)
    print("\n=== ROBUST ADVERSARIAL TRAINING RESULTS ===")
    print(res_df.to_string(index=False))
    res_df.to_csv(OUT / "robust_adversarial_results.csv", index=False)
    
    # Save the robust model for Streamlit
    joblib.dump(model, "models/robust_xgboost_phish_model.pkl")
    print("\n✅ Robust model saved to models/robust_xgboost_phish_model.pkl")

if __name__ == "__main__":
    main()