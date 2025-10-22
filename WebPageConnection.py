import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# --- App Title ---
st.title("📈 Share Market Prediction App (Pretrained Model)")

# --- Upload Dataset ---
uploaded_file = st.file_uploader("📤 Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.dataframe(df.head())
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

# --- Load Pretrained Model ---
model_file = st.file_uploader("🤖 Upload your trained model file (.pkl)", type=["pkl"])
if model_file is not None:
    try:
        model = pickle.load(model_file)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        st.stop()
else:
    st.warning("Please upload a pretrained model file.")
    st.stop()

# --- Feature & Target Selection ---
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns.")
    st.stop()

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

# --- Predict using pretrained model ---
try:
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    accuracy = r2 * 100

    st.success(f"🏆 Model Performance")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**Approx Accuracy:** {accuracy:.2f}%")

    # --- Plot ---
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
    <div style='text-align: center;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Pretrained Model Mode</p>
    </div>
    """,
    unsafe_allow_html=True
)
