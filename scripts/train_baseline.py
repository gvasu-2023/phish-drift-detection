"""Baseline: XGBoost on a random split."""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from phishing_detector.data.load import load_url_phish
from phishing_detector.calibration.metrics import (
    expected_calibration_error, brier, reliability_bins,
)

OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = load_url_phish()
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "ece": float(expected_calibration_error(y_test, proba, n_bins=15)),
        "brier": brier(y_test, proba),
    }
    print(json.dumps(metrics, indent=2))
    (OUT / "baseline_random_split.json").write_text(json.dumps(metrics, indent=2))
    bins = reliability_bins(y_test, proba, n_bins=15)
    (OUT / "baseline_reliability_bins.json").write_text(json.dumps(bins, indent=2))
    print(f"Wrote results to {OUT}")


if __name__ == "__main__":
    main()