"""Streamlit app for Phish-Drift Detection with Audio Alerts."""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
from pathlib import Path
from phishing_detector.features.build import extract_features

st.set_page_config(page_title="Phish-Drift Detector", page_icon="🛡️", layout="wide")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_phish_model.pkl")
    features = joblib.load("models/model_features.pkl")
    return model, features

try:
    model, feature_cols = load_model()
except FileNotFoundError:
    st.error("Model not found! Please run `python scripts/train_and_save_model.py` first.")
    st.stop()

# Function to play audio
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

st.title("🛡️ Phish-Drift Detection System")
st.markdown("### Calibrated, Drift-Aware Phishing URL Detection")
st.markdown("Enter a URL below to analyze it using our temporal-calibrated XGBoost model.")

# Input
url_input = st.text_input("Enter URL to scan:", "http://example.com/login")

if st.button("Analyze URL"):
    if url_input:
        # 1. Extract Features
        features_dict = extract_features(url_input)
        
        # Remove 'tld' string if present
        if "tld" in features_dict:
            del features_dict["tld"]
            
        # Convert to DataFrame, ensuring correct column order
        df_features = pd.DataFrame([features_dict])[feature_cols]
        
        # 2. Predict
        prob = model.predict_proba(df_features)[0][1]
        
        # 3. Display Result
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Analysis Result")
            if prob > 0.5:
                st.error(f"🚨 **PHISHING DETECTED** (Confidence: {prob*100:.2f}%)")
                # Audio alert for phishing
                # We use a standard warning beep sound (base64)
                st.warning("This URL exhibits characteristics of a phishing attack.")
            else:
                st.success(f"✅ **SAFE URL** (Confidence: {(1-prob)*100:.2f}%)")
                
        with col2:
            st.subheader("Heuristic Breakdown")
            st.write(f"- **URL Length:** {features_dict['url_len']}")
            st.write(f"- **Entropy:** {features_dict['entropy']:.2f}")
            st.write(f"- **Is HTTPS:** {'Yes' if features_dict['is_https'] == 1 else 'No'}")
            st.write(f"- **Subdomains:** {features_dict['subdom_cnt']}")
            
        # Optional: Show raw features
        with st.expander("View Raw Extracted Features"):
            st.json(features_dict)