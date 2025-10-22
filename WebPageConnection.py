import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
from sklearn.metrics import mean_squared_error, r2_score

st.title("📈 Share Market Prediction (Pretrained Model from GitHub)")

# --- Load Model from GitHub ---
MODEL_URL = "https://github.com/Sumiya-Ahasan/Share-market-project/blob/main/best_model.pkl"

try:
    st.info("📥 Downloading model from GitHub...")
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model = pickle.loads(response.content)
    st.success("✅ Model loaded successfully from GitHub!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Upload Dataset ---
uploaded_file = st.file_uploader("📤 Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.dataframe(df.head())
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

# --- Select Features & Target ---
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
target = st.selectbox("🎯 Select Target Variable", numeric_cols)
features = st.multiselect(
    "🧮 Select Input Features", [c for c in numeric_cols if c != target],
    default=[c for c in numeric_cols if c != target]
)

if len(features) == 0:
    st.error("Please select at least one feature.")
    st.stop()

X = df[features]
y = df[target]

# --- Predict and Evaluate ---
try:
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    st.success("🏆 Model Performance:")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**Accuracy:** {r2*100:.2f}%")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, color='blue', alpha=0.6)
    ax.plot(y, y, color='red')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("<div style='text-align:center;'>Developed with ❤️ by <b>Sumiya Ahasan</b></div>", unsafe_allow_html=True)
