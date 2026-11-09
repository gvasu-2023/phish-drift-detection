"""Train the Chaos Model against advanced evasion techniques."""
import pandas as pd
import numpy as np
import joblib
import random
import warnings
import sys
import os
from pathlib import Path
from xgboost import XGBClassifier
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from phishing_detector.features.build import extract_features

warnings.filterwarnings("ignore")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def generate_attacks(n=5000):
    attacks = []
    brands = ["google", "paypal", "amazon", "microsoft", "apple", "netflix", "facebook"]
    
    # 1. Typosquatting
    for _ in range(n // 4):
        brand = random.choice(brands)
        typo = brand.replace("o", "0").replace("l", "1").replace("e", "3").replace("m", "rn")
        attacks.append(f"http://{typo}{random.choice(['.com', '.net', '.xyz', '.info', '.biz'])}/login")

    # 2. Homoglyph / IDN
    homoglyphs = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441", "x": "\u0445", "y": "\u0443"}
    for _ in range(n // 4):
        brand = random.choice(brands)
        homo = "".join(homoglyphs.get(c, c) for c in brand)
        attacks.append(f"http://{homo}{random.choice(['.com', '.net'])}/login")

    # 3. Combosquatting (Semantic)
    words = ["secure", "login", "verify", "account", "update", "recovery", "wallet", "support"]
    for _ in range(n // 4):
        brand = random.choice(brands)
        w1, w2 = random.sample(words, 2)
        attacks.append(f"http://{brand}-{w1}-{w2}{random.choice(['.com', '.net', '.xyz', '.biz'])}/{w1}.php")

    # 4. ZWC and Obfuscated IPs
    for _ in range(n // 4):
        if random.random() < 0.5:
            brand = random.choice(brands)
            idx = random.randint(1, len(brand)-1)
            attacks.append(f"http://{brand[:idx]}\u200b{brand[idx:]}.com/login")
        else:
            ip = f"0x{random.randint(10,255):02x}.0x{random.randint(10,255):02x}.0x{random.randint(10,255):02x}.0x{random.randint(10,255):02x}"
            attacks.append(f"http://{ip}/admin.php")
    return attacks

def generate_benign(n=20000):
    benign = []
    words = ["blog", "article", "news", "help", "faq", "docs", "store", "product", "technology", "science", "review", "update", "about", "how-to", "guide", "review"]
    domains = ["medium", "wordpress", "github", "stackoverflow", "nytimes", "bbc", "shopify", "techcrunch", "wired", "forbes", "verge"]
    
    for _ in range(n):
        d = random.choice(domains)
        w1 = random.choice(words)
        w2 = random.choice(words)
        w3 = random.choice(words)
        year = random.randint(2018, 2026)
        tld = random.choice([".com", ".org", ".net", ".io", ".co"])
        
        # Generate highly complex, realistic blog URLs
        if random.random() < 0.5:
            benign.append(f"https://www.{d}-{w1}-{w2}-{year}{tld}/article/{w3}.html")
        else:
            benign.append(f"https://{d}-{w1}-{w2}{tld}/blog/{w3}-{year}-page")
    return benign

def main():
    print("1. Generating Advanced Evasion Data...")
    attack_urls = generate_attacks(5000)
    benign_urls = generate_benign(5000)
    
    df_targeted = pd.DataFrame({
        "url": attack_urls + benign_urls,
        "label": [1] * len(attack_urls) + [0] * len(benign_urls)
    })

    print("2. Loading Base Datasets...")
    df_base = pd.read_csv("data/processed/temporal_benchmark.csv")[["url", "label"]]
    
    try:
        df_phi = pd.read_csv("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
        url_col = next((c for c in df_phi.columns if c.lower() == 'url'), None)
        df_phi = df_phi[[url_col, 'label']].rename(columns={url_col: 'url'})
        df_phi['label'] = df_phi['label'].apply(lambda x: 1 if x == 0 else 0)
        df_base = pd.concat([df_base, df_phi.sample(n=20000, random_state=42)])
    except FileNotFoundError:
        pass

    print("3. Combining Base + Chaos Data...")
    df_combined = pd.concat([df_base, df_targeted]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Extracting features for {len(df_combined)} URLs...")
    X_list = []
    for u in tqdm(df_combined["url"], desc="Extracting"):
        feats = extract_features(str(u))
        feats.pop("tld", None)
        X_list.append(feats)
        
    X = pd.DataFrame(X_list)
    y = df_combined["label"].values

    print("4. Training the Chaos XGBoost Model...")
    model = XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X, y)

    print("\n5. Testing on advanced evasion URLs:")
    test_urls = [
        "http://paypa1.com/login", # Typosquatting
        "http://\u0430pple.com/login", # Homoglyph
        "http://amazon-support-verification.net/update", # Semantic
        "http://face\u200bbook.com/login", # ZWC
        "http://0xC0.0xA8.0x01.0x01/admin.php", # Obfuscated IP
        "https://medium-blog-about-technology-2026.com/article/ai.html" # Safe
    ]
    
    test_feats = []
    for u in test_urls:
        f = extract_features(u)
        f.pop("tld", None)
        test_feats.append(f)
        
    probs = model.predict_proba(pd.DataFrame(test_feats))[:, 1]
    
    for url, prob in zip(test_urls, probs):
        status = "PHISHING" if prob > 0.5 else "SAFE"
        print(f"   {status:9} ({prob*100:.2f}%) | {url}")

    joblib.dump(model, MODEL_DIR / "chaos_xgboost_model.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "model_features.pkl")
    print(f"\n✅ Chaos model saved.")

if __name__ == "__main__":
    main()