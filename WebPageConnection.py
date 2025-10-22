import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
from sklearn.metrics import mean_squared_error, r2_score

# --- App Title ---
st.title("📊 Share Market Prediction (Auto Model & Dataset from GitHub)")

# --- GitHub RAW Links ---
MODEL_URL = "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/best_model.pkl"
DATA_URL = "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/share_data.csv"  # তোমার dataset ফাইলের নাম অনুযায়ী বদলাও

# --- Load Model from GitHub ---
try:
    st.info("📥 Downloading pretrained model from GitHub...")
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model = pickle.loads(response.content)
    st.success("✅ Model loaded successfully from GitHub!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Load Dataset from GitHub ---
try:
    st.info("📊 Loading dataset from GitHub...")
    data_response = requests.get(DATA_URL)
    data_response.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(data_response.text))
    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# --- Select Target and Features ---
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns.")
    st.stop()

target = st.selectbox("🎯 Select Target Variable", numeric_cols, index=len(numeric_cols) - 1)
features = [c for c in numeric_cols if c != target]

X = df[features]
y = df[target]

# --- Predict using pretrained model ---
try:
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    st.success("🏆 Model Performance Summary")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**Accuracy:** {r2*100:.2f}%")

    # --- Plot Actual vs Predicted ---
    st.subheader("📉 Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, color='blue', alpha=0.6, label='Predicted')
    ax.plot(y, y, color='red', label='Actual')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted (Pretrained Model)")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Prediction failed: {e}")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Auto GitHub Integration</p>
    </div>
    """,
    unsafe_allow_html=True
)
