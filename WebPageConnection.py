import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import io
from sklearn.metrics import mean_squared_error, r2_score

# =============================
# --- App Title ---
# =============================
st.title("📊 Share Market Prediction ")

# --- GitHub RAW Link of Model ---
MODEL_URL = "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/best_model.pkl"

# --- Google Drive Direct Download Link ---
DATA_URL = "https://drive.google.com/uc?export=download&id=1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"

# =============================
# 🔹 Load Model from GitHub
# =============================
try:
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model = pickle.loads(response.content)
    model_name = model.__class__.__name__
    st.info(f"🧠 Using Model: **{model_name}**")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# =============================
# 🔹 Load Dataset from Google Drive
# =============================
try:
    data_response = requests.get(DATA_URL)
    data_response.raise_for_status()
    df = pd.read_csv(io.StringIO(data_response.text))
    st.dataframe(df.head())
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# =============================
# 🔹 Target & Feature Setup
# =============================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns.")
    st.stop()

target = st.selectbox("🎯 Select Target Variable", numeric_cols, index=len(numeric_cols) - 1)

# =============================
# 🔹 Align Dataset Features with Model
# =============================
try:
    if hasattr(model, "feature_names_in_"):
        required_features = list(model.feature_names_in_)
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            st.warning(f"⚠️ Missing columns detected: {missing}. Filling them with zeros.")
            for col in missing:
                df[col] = 0
        X = df[required_features]
    else:
        st.info("ℹ️ Model has no feature metadata; using all numeric columns.")
        X = df.select_dtypes(include=np.number)
except Exception as e:
    st.error(f"Feature alignment failed: {e}")
    st.stop()

# =============================
# 🔹 Prediction & Evaluation
# =============================
try:
    y_pred = model.predict(X)
    if target in df.columns:
        y = df[target]
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        accuracy = r2 * 100

        st.subheader("🏆 Model Performance Summary")
        st.write(f"**Model Used:** {model_name}")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**Accuracy:** {accuracy:.2f}%")

        # Plot Actual vs Predicted
        st.subheader("📉 Actual vs Predicted (Dataset)")
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, color='blue', alpha=0.6, label='Predicted')
        ax.plot(y, y, color='red', label='Actual')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Actual vs Predicted ({model_name})")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("ℹ️ Target not found in dataset, showing only predictions.")
        st.dataframe(pd.DataFrame({"Prediction": y_pred}))
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")

# =============================
# 🔹 Manual Input Prediction
# =============================
st.markdown("---")
st.subheader("🧮 Try Your Own Input")

try:
    # Identify input features from model
    if hasattr(model, "feature_names_in_"):
        input_features = list(model.feature_names_in_)
    else:
        input_features = [c for c in df.columns if c != target]

    user_input = {}
    cols = st.columns(2)
    for i, col_name in enumerate(input_features):
        with cols[i % 2]:
            val = st.number_input(
                f"{col_name}",
                value=float(df[col_name].mean()) if col_name in df.columns else 0.0
            )
            user_input[col_name] = val

    if st.button("🔮 Predict from Manual Input"):
        input_df = pd.DataFrame([user_input])
        pred_value = model.predict(input_df)[0]
        st.success(f"📈 **Predicted {target}: {pred_value:.2f}**")
        st.info(f"🧠 Model Used: **{model_name}**")

except Exception as e:
    st.error(f"⚠️ Manual input prediction failed: {e}")

# =============================
# 🔹 Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <p>Developed with  by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Auto Model + Google Drive + Manual Input</p>
    </div>
    """,
    unsafe_allow_html=True
)
