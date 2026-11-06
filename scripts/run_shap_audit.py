"""SHAP Feature Stability Audit: Compare feature importance Past vs Future."""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")
FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading temporal dataset...")
    df = pd.read_csv("data/processed/temporal_benchmark.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', utc=True)
    df["year"] = df["timestamp"].dt.year
    
    # Fix Tranco dates
    phish_years = df[df["label"] == 1]["year"]
    min_year, max_year = phish_years.min(), phish_years.max()
    np.random.seed(42)
    benign_mask = df["label"] == 0
    df.loc[benign_mask, "year"] = np.random.randint(min_year, max_year, size=benign_mask.sum())

    # Split
    SPLIT_YEAR = 2024
    train_df = df[df["year"] < SPLIT_YEAR]
    test_df = df[df["year"] >= SPLIT_YEAR]
    
    # Balance (using 3000 to keep SHAP calculation fast)
    def get_balanced_set(subset, max_n=3000):
        pos = subset[subset["label"] == 1]
        neg = subset[subset["label"] == 0]
        n = min(max_n, len(pos), len(neg))
        return pd.concat([pos.sample(n=n, random_state=42), neg.sample(n=n, random_state=42)]).sample(frac=1, random_state=42)
        
    train_bal = get_balanced_set(train_df)
    test_bal = get_balanced_set(test_df)

    drop_cols = ["url", "timestamp", "year", "label", "tld"]
    X_train = train_bal.drop(columns=[c for c in drop_cols if c in train_bal.columns])
    y_train = train_bal["label"].values
    X_test = test_bal.drop(columns=[c for c in drop_cols if c in test_bal.columns])

    print("\nTraining XGBoost...")
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    print("\nCalculating SHAP values for Past and Future...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (using a sample to save memory)
    shap_past = explainer.shap_values(X_train.sample(1000, random_state=42))
    shap_future = explainer.shap_values(X_test.sample(1000, random_state=42))

    # Extract mean absolute SHAP values for each feature
    imp_past = np.abs(shap_past).mean(axis=0)
    imp_future = np.abs(shap_future).mean(axis=0)
    
    features = X_train.columns
    imp_df = pd.DataFrame({
        "Feature": features,
        "Past_Importance": imp_past,
        "Future_Importance": imp_future
    })
    
    # Calculate a "Drift Score" (Ratio of Future to Past)
    imp_df["Drift_Ratio"] = imp_df["Future_Importance"] / (imp_df["Past_Importance"] + 1e-9)
    imp_df = imp_df.sort_values(by="Past_Importance", ascending=False).head(15)
    
    print("\nTop 15 Features by Importance:")
    print(imp_df[["Feature", "Past_Importance", "Future_Importance", "Drift_Ratio"]].to_string(index=False))
    
    # Save table
    imp_df.to_csv("results/tables/shap_feature_drift.csv", index=False)

    # Plotting the comparison
    print("\nGenerating Feature Drift Plot...")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(imp_df))
    width = 0.35
    
    plt.bar(x - width/2, imp_df["Past_Importance"], width, label='Past (<2024)', color='#4C72B0')
    plt.bar(x + width/2, imp_df["Future_Importance"], width, label='Future (>=2024)', color='#C44E52')
    
    plt.xticks(x, imp_df["Feature"], rotation=45, ha='right')
    plt.ylabel("Mean |SHAP value| (Impact on Model Output)", fontsize=12)
    plt.title("SHAP Feature Importance: Past vs Future Concept Drift", fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    plot_path = FIG / "shap_feature_drift.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Plot saved to {plot_path}")

if __name__ == "__main__":
    main()