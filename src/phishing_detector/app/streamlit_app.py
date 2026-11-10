"""Streamlit app for Phish-Drift Detection with Explainable AI and Batch Processing."""
import sys
import os
import io

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from phishing_detector.features.build import extract_features
from phishing_detector.features.real_world import analyze_real_world

st.set_page_config(page_title="Phish-Drift Detector", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    model_path = os.path.join(src_dir, "..", "models", "production_xgboost_model.pkl")
    features_path = os.path.join(src_dir, "..", "models", "model_features.pkl")
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return None, None
    return joblib.load(model_path), joblib.load(features_path)

model, feature_cols = load_model()

if model is None:
    st.error("Model files not found. Please run `python scripts/auto_update_pipeline.py` first.")
    st.stop()

st.title("🛡️ Phish-Drift Enterprise Detection System")
st.markdown("### Calibrated, Drift-Aware Phishing URL Detection with Threat Intel")
st.markdown("Enter URLs manually OR upload a CSV/TXT file containing a list of URLs.")

# --- UI IMPROVEMENT: Batch Upload ---
col1, col2 = st.columns([2, 1])

with col1:
    url_input = st.text_area(
        "Enter URLs to scan (one per line):", 
        "https://chat.deepseek.com\nhttps://www.perplexity.ai/\nhttp://paypal-secure-login.xyz/update-account.php\nhttp://0xC0.0xA8.0x01.0x01/admin.php",
        height=150
    )

with col2:
    uploaded_file = st.file_uploader("Or upload a file (.txt or .csv)", type=["txt", "csv"])
    st.info("Upload a text file with one URL per line, or a CSV with a 'url' column.")

if st.button("Analyze URLs", type="primary"):
    urls = []
    
    # Parse Inputs
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
            col_name = next((c for c in df_upload.columns if 'url' in c.lower()), df_upload.columns[0])
            urls = df_upload[col_name].dropna().astype(str).tolist()
        else:
            content = uploaded_file.getvalue().decode("utf-8")
            urls = [line.strip() for line in content.split('\n') if line.strip()]
    elif url_input:
        urls = [line.strip() for line in url_input.split('\n') if line.strip()]

    if urls:
        results = []
        progress_bar = st.progress(0)
        
        for i, url in enumerate(urls):
            try:
                # 1. Real World Threat Intel
                intel = analyze_real_world(url)
                scan_url = intel["final_url"]
                
                # 2. Extract Features & ML Predict
                features_dict = extract_features(scan_url)
                if "tld" in features_dict:
                    del features_dict["tld"]
                    
                df_features = pd.DataFrame([features_dict])[feature_cols]
                ml_prob = model.predict_proba(df_features)[0][1]
                
                # 3. Combine ML + Intel
                final_prob = max(0.0, min(1.0, ml_prob + intel["risk_score_adjustment"]))
                status = "🚨 PHISHING" if final_prob > 0.5 else "✅ SAFE"
                confidence = final_prob if final_prob > 0.5 else (1 - final_prob)
                
                # 4. UI IMPROVEMENT: Explainable AI Reasons
                reason_str = ", ".join(intel["reasons"]) if intel["reasons"] else "Lexical features only"
                if not intel["reasons"] and final_prob > 0.5:
                    reason_str = "High-risk lexical structure (Entropy, Subdomains, etc.)"

                results.append({
                    "URL": url,
                    "Prediction": status,
                    "Confidence": f"{confidence*100:.2f}%",
                    "Domain Age (Days)": intel['domain_age_days'] if intel['domain_age_days'] != -1 else "Unknown",
                    "Explainability (Why?)": reason_str
                })
            except Exception as e:
                results.append({"URL": url, "Prediction": "❌ ERROR", "Confidence": "-", "Domain Age (Days)": "-", "Explainability (Why?)": str(e)})
                
            progress_bar.progress((i + 1) / len(urls))
            
        st.subheader("Analysis Results")
        df_results = pd.DataFrame(results)
        
        def highlight_status(val):
            color = '#ffcccc' if 'PHISHING' in val else '#ccffcc'
            if 'ERROR' in val: color = '#eeeeee'
            return f'background-color: {color}'
            
        st.dataframe(df_results.style.map(highlight_status, subset=['Prediction']), use_container_width=True)
        
        phish_count = sum(1 for r in results if "PHISHING" in r["Prediction"])
        st.warning(f"⚠️ Detected **{phish_count}** malicious URLs out of **{len(urls)}** scanned.")
        
        # 5. UI IMPROVEMENT: Download Results
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv,
            file_name='phish_drift_report.csv',
            mime='text/csv',
        )