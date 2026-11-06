"""Tier 3: Adversarial Robustness Evaluation."""
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
import warnings

from phishing_detector.calibration.metrics import expected_calibration_error, reliability_bins
from phishing_detector.features.build import extract_features

warnings.filterwarnings("ignore")
FIG = Path("results/figures")
OUT = Path("results/tables")
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

def perturb_url(url, attack_type):
    """Apply adversarial mutations to a URL."""
    try:
        # Ensure it has a scheme
        if not url.startswith("http"):
            url = "http://" + url
            
        parts = url.split("://")
        scheme, rest = parts[0], parts[1]
        
        if attack_type == "typosquatting":
            # Replace o->0, l->1, e->3 to trick lexical models
            rest = rest.replace("o", "0").replace("l", "1").replace("e", "3")
            return f"{scheme}://{rest}"
            
        elif attack_type == "subdomain_padding":
            # Add random subdomains: secure.login.random.com
            rand_sub = ''.join(random.choices(string.ascii_lowercase, k=5))
            return f"{scheme}://{rand_sub}.secure.{rest}"
            
        elif attack_type == "path_obfuscation":
            # Add long random paths to increase entropy and slash count
            rand_path = ''.join(random.choices(string.ascii_letters + string.digits, k=30))
            return f"{scheme}://{rest}/{rand_path}/login.php"
            
        return url
    except:
        return url

def main():
    print("Loading temporal dataset and models...")
    df = pd.read_csv("data/processed/temporal_benchmark.csv")
    
    # We take a sample of 1000 phishing and 1000 benign for testing
    test_pos = df[df["label"] == 1].sample(n=1000, random_state=42)
    test_neg = df[df["label"] == 0].sample(n=1000, random_state=42)
    test_clean = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Load the trained XGBoost model
    model = joblib.load("models/xgboost_phish_model.pkl")
    feature_cols = joblib.load("models/model_features.pkl")

    # Prepare Calibration data (using 20% of clean data)
    train_cal, test_eval = train_test_split(test_clean, test_size=0.8, random_state=42, stratify=test_clean["label"])
    
    # Fit the calibrator on clean data
    X_cal = pd.DataFrame([extract_features(u) for u in train_cal["url"]]).drop(columns=["tld"], errors="ignore")[feature_cols]
    y_cal = train_cal["label"].values
    
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_cal, y_cal)

    # Attack types
    attacks = ["clean", "typosquatting", "subdomain_padding", "path_obfuscation"]
    results = []

    for attack in attacks:
        print(f"\nEvaluating Attack: {attack.upper()}")
        
        # Apply perturbations if not clean
        if attack == "clean":
            eval_urls = test_eval["url"].tolist()
        else:
            eval_urls = [perturb_url(u, attack) for u in test_eval["url"]]
            
        # Extract features
        X_eval = pd.DataFrame([extract_features(u) for u in eval_urls]).drop(columns=["tld"], errors="ignore")[feature_cols]
        y_eval = test_eval["label"].values
        
        # Predict Raw
        proba_raw = model.predict_proba(X_eval)[:, 1]
        acc_raw = accuracy_score(y_eval, (proba_raw >= 0.5).astype(int))
        ece_raw = expected_calibration_error(y_eval, proba_raw, n_bins=15)
        
        # Predict Calibrated
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
    print("\n=== ADVERSARIAL ROBUSTNESS RESULTS ===")
    print(res_df.to_string(index=False))
    
    res_df.to_csv(OUT / "adversarial_robustness.csv", index=False)
    
    # Plotting the attack impact
    plt.figure(figsize=(8, 5))
    plt.bar(res_df["Attack Type"], res_df["ECE (Raw)"].astype(float), color='#C44E52', label='Raw Model (Uncalibrated)')
    plt.bar(res_df["Attack Type"], res_df["ECE (Cal) " if "ECE (Cal) " in res_df else "ECE (Cal)"].astype(float), color='#55A868', label='Calibrated Model')
    plt.ylabel("Expected Calibration Error (ECE)", fontsize=12)
    plt.title("Impact of Adversarial Mutations on Model Calibration", fontsize=14)
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    
    plot_path = FIG / "adversarial_robustness.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ Figure saved to {plot_path}")
    print(f"✅ Table saved to {OUT / 'adversarial_robustness.csv'}")

if __name__ == "__main__":
    main()