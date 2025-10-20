import os
import joblib
import urllib.request
import streamlit as st

MODEL_URL  = "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/best_model.pkl"
MODEL_PATH = "best_model.pkl"

if not os.path.exists(MODEL_PATH):
    try:
        st.info("📦 Downloading the model from GitHub…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to download model: {e}")
        st.stop()

try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()
