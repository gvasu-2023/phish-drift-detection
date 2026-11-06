# Title: Calibrated, Drift-Aware Lightweight Adaptation for Phishing URL Detection

## Abstract
- Problem: Phishing URL detectors degrade in real-world deployment due to concept drift and prior probability shift.
- Solution: We propose a lightweight calibration framework using Isotonic Regression.
- Results: Our method reduces Expected Calibration Error (ECE) from X to Y without retraining the model.

## I. Introduction
- Phishing is a major cybersecurity threat.
- ML models achieve 99% accuracy in lab settings but fail in production due to time drift.
- Contribution: We evaluate temporal drift on 150K URLs and propose a warm-up calibration fix.

## II. Related Work
- URL-Phish dataset (Kaggle).
- Concept drift in cybersecurity.
- Calibration metrics (ECE, Brier Score).

## III. Methodology
- **Dataset Construction:** PhishTank + Tranco (150K URLs, 22 features).
- **Temporal Splitting:** Train on <2024, Test on >=2024.
- **Model:** XGBoost.
- **Lightweight Adaptation:** Isotonic Regression on a 10% warm-up set.

## IV. Experimental Results
- **Table 1:** Baseline vs Temporal Split (show accuracy drop or ECE increase).
- **Figure 1:** Reliability Diagram (Unadapted vs Adapted).
- **Figure 2:** SHAP Feature Drift (proving feature stability over time).
- **Table 2:** ECE Before vs After Isotonic Calibration.

## V. Conclusion
- Lexical features are stable, but prior shift causes miscalibration.
- Lightweight calibration fixes this without expensive retraining.