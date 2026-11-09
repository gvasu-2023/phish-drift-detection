# Phish-Drift Detection

A calibrated, drift-aware evaluation framework for temporal phishing URL detection.

## Research Question
How do accuracy, calibration, and feature importance of phishing URL classifiers degrade under temporal distribution shift, adversarial evasion, and cross-dataset domain shift, and which lightweight adaptation strategies recover them?

## Project Status
- [x] Temporal Dataset Construction (149K URLs)
- [x] Lexical & DistilBERT Feature Extraction
- [x] Baseline Models (XGBoost, Random Forest, MLP, LR)
- [x] Temporal Split & Calibration Analysis (Isotonic Regression)
- [x] SHAP Feature-Stability Audit
- [x] Adversarial Training (Fixed 50% collapse to 99.7%)
- [x] Multi-Year Drift Analysis
- [x] Cross-Dataset Generalization (OpenPhish & PhiUSIIL)
- [x] Domain Adaptation (Combined Training 99.66% accuracy)
- [x] PSI Drift Detection
- [x] Streamlit Cloud Deployment

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place raw data in `data/raw/`
3. Run the full pipeline: `python scripts/run_all.py`
4. Launch the app: `streamlit run src/phishing_detector/app/streamlit_app.py`