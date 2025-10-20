import streamlit as st
import numpy as np
import joblib
import os
import urllib.request

# --- App Title ---
st.title("📈 Share Market Next Day Prediction")
st.write("Enter the last 4 days' closing prices to predict the next day's closing price using the trained model.")

# --- Model File from GitHub ---
MODEL_URL = "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/best_model.pkl"
MODEL_PATH = "best_model.pkl"

# --- Download Model if not Found ---
if not os.path.exists(MODEL_PATH):
    try:
        st.info("📦 Downloading trained model from GitHub...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to download model: {e}")
        st.stop()

# --- Load Model ---
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- User Input Section ---
st.subheader("🧮 Input Last 4 Days' Closing Prices")

close1 = st.number_input("Close Price - Day 1", value=0.0, format="%.2f")
close2 = st.number_input("Close Price - Day 2", value=0.0, format="%.2f")
close3 = st.number_input("Close Price - Day 3", value=0.0, format="%.2f")
close4 = st.number_input("Close Price - Day 4", value=0.0, format="%.2f")

# Combine inputs into array for prediction
features = np.array([[close1, close2, close3, close4]])

# --- Predict Button ---
if st.button("🔮 Predict Next Day Close Price"):
    try:
        prediction = model.predict(features)[0]
        st.success(f"💹 Predicted Next Day Close Price: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Powered by Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
