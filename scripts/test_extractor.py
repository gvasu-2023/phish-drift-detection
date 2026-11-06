"""Verify our feature extractor matches the Kaggle dataset."""
import pandas as pd
from phishing_detector.features.build import extract_features

def main():
    print("Loading dataset...")
    df = pd.read_csv("data/raw/url_phish.csv", nrows=5000)
    
    print("Testing our extractor on 5 random URLs...")
    sample = df.sample(5, random_state=42)
    
    for _, row in sample.iterrows():
        url = row["url"]
        our_feats = extract_features(url)
        
        # Compare a few key features
        print(f"\nURL: {url[:60]}...")
        print(f"Length     | Kaggle: {row['url_len']:3} | Ours: {our_feats['url_len']:3}")
        print(f"Entropy    | Kaggle: {row['entropy']:.4f} | Ours: {our_feats['entropy']:.4f}")
        print(f"Is IP      | Kaggle: {row['is_ip']:3} | Ours: {our_feats['is_ip']:3}")
        print(f"Path Len   | Kaggle: {row['path_len']:3} | Ours: {our_feats['path_len']:3}")

if __name__ == "__main__":
    main()