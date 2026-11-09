"""Train a targeted model against Typosquatting, Brand Impersonation, and IP Obfuscation."""
import pandas as pd
import numpy as np
import joblib
import random
import string
from pathlib import Path
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from phishing_detector.features.build import extract_features

warnings.filterwarnings("ignore")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def generate_attacks(n=3000):
    """Generate synthetic URLs for the 3 specific attack types."""
    attacks = []
    
    # 1. Typosquatting (Homoglyphs)
    brands = ["google", "paypal", "amazon", "microsoft", "apple", "netflix", "facebook"]
    for _ in range(n // 3):
        brand = random.choice(brands)
        typo = brand.replace("o", "0").replace("l", "1").replace("e", "3").replace("m", "rn").replace("i", "1")
        tld = random.choice([".com", ".net", ".xyz", ".info", ".biz"])
        attacks.append(f"http://{typo}{tld}/login")
        
    # 2. Brand Impersonation & Semantic Tricks
    words = ["secure", "login", "verify", "account", "update", "recovery", "wallet", "seed-phrase", "support"]
    for _ in range(n // 3):
        brand = random.choice(brands)
        w1, w2 = random.sample(words, 2)
        tld = random.choice([".com", ".net", ".xyz", ".info", ".biz", ".online"])
        attacks.append(f"http://{brand}-{w1}-{w2}{tld}/{w1}.php")
        
    # 3. IP & Obfuscated Links
    for _ in range(n // 3):
        # Standard IP
        ip = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
        # Hex IP
        hex_ip = f"0x{random.randint(10,255):02x}.0x{random.randint(10,255):02x}.0x{random.randint(10,255):02x}.0x{random.randint(10,255):02x}"
        path = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        tld = random.choice(["", ".com", ".net"])
        
        ip_choice = random.choice([ip, hex_ip])
        attacks.append(f"http://{ip_choice}{tld}/{path}/login.php")
        
    return attacks

def generate_benign(n=3000):
    """Generate benign URLs with similar structures to the attacks (Hard Negatives)."""
    benign = []
    words = ["blog", "article", "news", "help", "faq", "docs", "store", "product"]
    domains = ["medium", "wordpress", "github", "stackoverflow", "nytimes", "bbc", "shopify", "amazon"]
    
    for _ in range(n):
        d = random.choice(domains)
        w1, w2 = random.sample(words, 2)
        tld = random.choice([".com", ".org", ".net"])
        benign.append(f"https://www.{d}{tld}/{w1}/{w2}-page.html")
        
    return benign

def main():
    print("1. Generating Targeted Attack Data...")
    attack_urls = generate_attacks(3000)
    benign_urls = generate_benign(3000)
    
    df_targeted = pd.DataFrame({
        "url": attack_urls + benign_urls,
        "label": [1] * len(attack_urls) + [0] * len(benign_urls)
    })

    print("2. Loading Base Datasets (Temporal + PhiUSIIL)...")
    df_base = pd.read_csv("data/processed/temporal_benchmark.csv")[["url", "label"]]
    
    # 3. Combine everything
    print("3. Combining Base + Targeted Data...")
    df_combined = pd.concat([df_base, df_targeted]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Extracting features for {len(df_combined)} URLs...")
    X_list = []
    for u in tqdm(df_combined["url"], desc="Extracting"):
        feats = extract_features(str(u))
        feats.pop("tld", None)
        X_list.append(feats)
        
    X = pd.DataFrame(X_list)
    y = df_combined["label"].values

    print("4. Training the Enterprise XGBoost Model...")
    model = XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X, y)

    # Test on your exact failing URLs!
    print("\n5. Testing on your failing URLs:")
    test_urls = [
        "http://metamask-wallet-recovery-seed-phrase.biz",
        "http://dynamic-dns.net",
        "https://amaz0n-support-portal-orders-verification.net",
        "http://0x58.0xCC.0xCA.0x63/paypal/"
    ]
    
    test_feats = []
    for u in test_urls:
        f = extract_features(u)
        f.pop("tld", None)
        test_feats.append(f)
        
    X_test = pd.DataFrame(test_feats)
    probs = model.predict_proba(X_test)[:, 1]
    
    for url, prob in zip(test_urls, probs):
        status = "PHISHING" if prob > 0.5 else "SAFE"
        print(f"   {status:9} ({prob*100:.2f}%) | {url}")

    # Save model
    joblib.dump(model, MODEL_DIR / "enterprise_xgboost_model.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "model_features.pkl")
    print(f"\n✅ Enterprise model saved to {MODEL_DIR / 'enterprise_xgboost_model.pkl'}")

if __name__ == "__main__":
    main()