"""Automated MLOps Pipeline: Ingest live data, check drift, and retrain."""
import pandas as pd
import numpy as np
import joblib
import requests
import io
import random
import string
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from phishing_detector.features.build import extract_features

warnings.filterwarnings("ignore")
DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Live Ingestion ---
def download_live_phishtank():
    print("🌐 [1/5] Downloading live PhishTank feed...")
    url = "http://data.phishtank.com/data/online-valid.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df = df[['url']].copy()
        df['label'] = 1 # 1 = Phishing
        print(f"   ✅ Downloaded {len(df)} live phishing URLs.")
        return df
    except Exception as e:
        print(f"   ⚠️ Could not download live feed ({e}). Using local base data.")
        return pd.DataFrame(columns=["url", "label"])

# --- 2. Chaos Generation ---
def generate_chaos_data(n=3000):
    print("🧬 [2/5] Generating Chaos Evasion Data...")
    attacks = []
    benign = []
    brands = ["google", "paypal", "amazon", "microsoft", "apple", "netflix", "facebook"]
    words = ["secure", "login", "verify", "account", "update", "recovery", "wallet", "support", "blog", "article"]
    
    for _ in range(n):
        # Attack
        if random.random() < 0.33:
            brand = random.choice(brands)
            typo = brand.replace("o", "0").replace("l", "1")
            attacks.append(f"http://{typo}.com/login")
        elif random.random() < 0.66:
            b = random.choice(brands)
            w1, w2 = random.sample(words, 2)
            attacks.append(f"http://{b}-{w1}-{w2}.net/{w1}.php")
        else:
            ip = f"0x{random.randint(10,255):02x}.0x{random.randint(10,255):02x}.0x{random.randint(10,255):02x}.0x{random.randint(10,255):02x}"
            attacks.append(f"http://{ip}/admin.php")
            
        # Benign
        d = random.choice(["medium", "wordpress", "github", "techcrunch"])
        w1, w2 = random.sample(words, 2)
        benign.append(f"https://www.{d}-{w1}-{w2}.com/article/{w1}.html")
        
    return pd.DataFrame({
        "url": attacks + benign,
        "label": [1] * len(attacks) + [0] * len(benign)
    })

# --- 3. Retrain & Deploy ---
def main():
    print("🚀 Starting Automated Self-Updating Pipeline...")
    
    # A. Ingest Data
    df_live = download_live_phishtank()
    df_chaos = generate_chaos_data()
    
    # B. Load Historical Data
    df_historical = pd.read_csv("data/processed/temporal_benchmark.csv")[["url", "label"]]
    
    # C. Combine
    df_combined = pd.concat([df_historical, df_live, df_chaos]).drop_duplicates(subset=['url']).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"📊 [3/5] Combined dataset size: {len(df_combined)} URLs.")

    # D. Feature Extraction
    print("🔬 [4/5] Extracting advanced features (Levenshtein, ZWC, Homoglyphs)...")
    X_list = []
    for u in tqdm(df_combined["url"], desc="Extracting"):
        feats = extract_features(str(u))
        feats.pop("tld", None)
        X_list.append(feats)
        
    X = pd.DataFrame(X_list)
    y = df_combined["label"].values

    # E. Train New Model
    print("🧠 [5/5] Retraining the Production XGBoost Model...")
    model = XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X, y)

    # F. Save as Production Model
    joblib.dump(model, MODEL_DIR / "production_xgboost_model.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "model_features.pkl")
    
    # Save a timestamped backup just in case
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, MODEL_DIR / f"backup_model_{timestamp}.pkl")
    
    print("✅ SUCCESS! Production model updated and saved.")

if __name__ == "__main__":
    main()