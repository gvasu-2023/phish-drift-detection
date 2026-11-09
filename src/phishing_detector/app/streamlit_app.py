"""Streamlit app for Phish-Drift Detection with Bulk Real-time Processing."""
import sys
import os

# Calculate the absolute path to the ROOT directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
src_dir = os.path.join(ROOT_DIR, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from phishing_detector.features.build import extract_features

st.set_page_config(page_title="Phish-Drift Detector", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    model_path = os.path.join(ROOT_DIR, "models", "combined_xgboost_phish_model.pkl")
    features_path = os.path.join(ROOT_DIR, "models", "model_features.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return None, None
        
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

model, feature_cols = load_model()

if model is None:
    st.error("Model files not found in the repository! Please ensure the .pkl files are pushed to GitHub.")
    st.stop()

st.title("🛡️ Phish-Drift Real-Time Detection System")
st.markdown("### Calibrated, Drift-Aware Phishing URL Detection")
st.markdown("Paste one or multiple URLs below to analyze them in real-time using our Combined Robust XGBoost model.")

# Text area for multiple URLs
url_input = st.text_area(
    "Enter URLs to scan (one per line):", 
    "http://google.com\nhttp://paypal-secure-login.xyz/update-account.php\nhttp://karunya.edu.in",
    height=150
)

if st.button("Analyze URLs"):
    if url_input:
        urls = [line.strip() for line in url_input.split('\n') if line.strip()]
        
        results = []
        progress_bar = st.progress(0)
        
        for i, url in enumerate(urls):
            # 1. Extract Features
            features_dict = extract_features(url)
            if "tld" in features_dict:
                del features_dict["tld"]
                
            df_features = pd.DataFrame([features_dict])[feature_cols]
            
            # 2. Predict
            prob = model.predict_proba(df_features)[0][1]
            
            # 3. Format Result
            status = "🚨 PHISHING" if prob > 0.5 else "✅ SAFE"
            confidence = prob if prob > 0.5 else (1 - prob)
            
            results.append({
                "URL": url,
                "Prediction": status,
                "Confidence": f"{confidence*100:.2f}%",
                "Entropy": round(features_dict['entropy'], 2),
                "Length": features_dict['url_len'],
                "HTTPS": "Yes" if features_dict['is_https'] == 1 else "No"
            })
            
            progress_bar.progress((i + 1) / len(urls))
            
        # Display Results Table
        st.subheader("Analysis Results")
        df_results = pd.DataFrame(results)
        
        # Color coding the table
        def highlight_status(val):
            color = '#ffcccc' if 'PHISHING' in val else '#ccffcc'
            return f'background-color: {color}'
            
        st.dataframe(df_results.style.map(highlight_status, subset=['Prediction']), use_container_width=True)
        
        # Summary Metrics
        phish_count = sum(1 for r in results if "PHISHING" in r["Prediction"])
        st.warning(f"⚠️ Detected **{phish_count}** malicious URLs out of **{len(urls)}** scanned.")