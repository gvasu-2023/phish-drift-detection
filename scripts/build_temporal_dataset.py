"""Build a time-stamped dataset from PhishTank and Tranco."""
import zipfile
import requests
import pandas as pd
import io
from pathlib import Path
from tqdm import tqdm
from phishing_detector.features.build import extract_features

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def download_tranco():
    print("Downloading Tranco Top 100k benign domains...")
    url = "https://tranco-list.eu/top-1m.csv.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_filename = z.namelist()[0]
    
    df = pd.read_csv(z.open(csv_filename), names=["rank", "domain"])
    # Keep top 100,000 to balance the dataset
    df = df.head(100000)
    import random
    # Randomly add http or https to prevent data leakage
    df["url"] = df["domain"].apply(lambda x: f"{random.choice(['http', 'https'])}://{x}")
    # Tranco is updated daily; we will use today's date for these
    df["timestamp"] = pd.Timestamp.now().normalize()
    df["label"] = 0  # Benign
    
    print(f"Downloaded {len(df)} benign URLs.")
    return df

def load_phishtank():
    path = RAW_DIR / "phishtank.csv"
    if not path.exists():
        print(f"⚠️ WARNING: {path} not found.")
        print("Please download a PhishTank CSV with timestamps from Kaggle and save it here.")
        return None
    
    df = pd.read_csv(path)
    # Try to find the url column and timestamp column dynamically
    url_col = next((c for c in df.columns if "url" in c.lower()), None)
    time_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    
    if not url_col or not time_col:
        print(f"⚠️ Could not find URL or Time column in {path}.")
        print(f"Columns available: {list(df.columns)}")
        return None
        
    print(f"Found URL column: '{url_col}', Time column: '{time_col}'")
    
    # Keep only what we need, rename, and convert time
    df = df[[url_col, time_col]].copy()
    df.columns = ["url", "timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Drop rows with bad timestamps or URLs
    df = df.dropna(subset=["timestamp", "url"])
    
    # Optional: Filter to a specific year if needed, but we'll keep all for now
    df["label"] = 1  # Phishing
    
    print(f"Loaded {len(df)} phishing URLs.")
    return df

def main():
    # 1. Load PhishTank
    phish_df = load_phishtank()
    
    # 2. Load Tranco
    benign_df = download_tranco()
    
    if phish_df is not None:
        print("Combining datasets...")
        combined = pd.concat([phish_df[["url", "timestamp", "label"]], 
                              benign_df[["url", "timestamp", "label"]]])
    else:
        print("Proceeding with Tranco only (testing mode)...")
        combined = benign_df[["url", "timestamp", "label"]]
        
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nExtracting features for {len(combined)} URLs. This will take a few minutes...")
    
    # 3. Extract Features
    features_list = []
    for url in tqdm(combined["url"], desc="Extracting"):
        features_list.append(extract_features(url))
        
    features_df = pd.DataFrame(features_list)
    
    # Drop the tld string column as XGBoost needs numeric only
    if "tld" in features_df.columns:
        features_df = features_df.drop(columns=["tld"])
        
    # 4. Combine back with metadata
    final_df = pd.concat([combined[["url", "timestamp", "label"]], features_df], axis=1)
    
    out_path = PROCESSED_DIR / "temporal_benchmark.csv"
    final_df.to_csv(out_path, index=False)
    
    print(f"\n✅ Success! Saved time-stamped dataset to {out_path}")
    print(final_df["timestamp"].describe())

if __name__ == "__main__":
    main()