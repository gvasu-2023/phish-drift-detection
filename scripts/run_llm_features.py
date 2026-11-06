"""Tier 2: Extract DistilBERT semantic embeddings and evaluate temporal drift."""
import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

from phishing_detector.calibration.metrics import expected_calibration_error

warnings.filterwarnings("ignore")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

# Check for GPU (if you have one), otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_embeddings(urls, tokenizer, model, batch_size=32):
    """Extract [CLS] embeddings from DistilBERT."""
    embeddings = []
    # Disable gradient calculation for inference speed
    with torch.no_grad():
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size].tolist()
            
            # Tokenize
            inputs = tokenizer(
                batch_urls, padding=True, truncation=True, 
                max_length=128, return_tensors="pt"
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Extract the [CLS] token (first token) from the last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
    return np.vstack(embeddings)

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
    
    # 1. Sample data (to keep laptop inference fast)
    train_pos = train_df[train_df["label"] == 1].sample(n=2500, random_state=42)
    train_neg = train_df[train_df["label"] == 0].sample(n=2500, random_state=42)
    train_bal = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

    test_pos = test_df[test_df["label"] == 1].sample(n=1000, random_state=42)
    test_neg = test_df[test_df["label"] == 0].sample(n=1000, random_state=42)
    test_bal = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

    print("Loading DistilBERT model from HuggingFace...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)
    bert_model.eval() # Set to evaluation mode

    print("\nExtracting LLM embeddings for Train data (This may take 1-2 minutes)...")
    X_train_emb = get_embeddings(train_bal["url"], tokenizer, bert_model)
    y_train = train_bal["label"].values

    print("Extracting LLM embeddings for Test data...")
    X_test_emb = get_embeddings(test_bal["url"], tokenizer, bert_model)
    y_test = test_bal["label"].values

    # 2. Train Classifier (Logistic Regression works best on dense embeddings)
    print("\nTraining Logistic Regression on LLM Embeddings...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_emb, y_train)

    # 3. Evaluate Raw
    proba_raw = clf.predict_proba(X_test_emb)[:, 1]
    acc_raw = accuracy_score(y_test, (proba_raw >= 0.5).astype(int))
    ece_raw = expected_calibration_error(y_test, proba_raw, n_bins=15)

    # 4. Apply Isotonic Calibration
    print("Applying Isotonic Calibration...")
    X_warmup, X_eval, y_warmup, y_eval = train_test_split(
        X_test_emb, y_test, test_size=0.8, random_state=42, stratify=y_test
    )
    
    calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
    calibrated_clf.fit(X_warmup, y_warmup)
    
    proba_cal = calibrated_clf.predict_proba(X_eval)[:, 1]
    acc_cal = accuracy_score(y_eval, (proba_cal >= 0.5).astype(int))
    ece_cal = expected_calibration_error(y_eval, proba_cal, n_bins=15)

    # 5. Display Results
    results = [{
        "Model": "DistilBERT (LLM)",
        "Acc (Raw)": f"{acc_raw:.4f}",
        "ECE (Raw)": f"{ece_raw:.4f}",
        "Acc (Cal)": f"{acc_cal:.4f}",
        "ECE (Cal)": f"{ece_cal:.4f}"
    }]
    
    res_df = pd.DataFrame(results)
    print("\n=== LLM TEMPORAL EVALUATION ===")
    print(res_df.to_string(index=False))
    
    res_df.to_csv(OUT / "llm_comparison.csv", index=False)
    print(f"\n✅ Table saved to {OUT / 'llm_comparison.csv'}")

if __name__ == "__main__":
    main()