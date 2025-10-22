import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import io
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Share Market Prediction", page_icon="📊", layout="wide")
st.title("📊 Share Market Prediction App (Best Model)")

# =============================
# 🔹 Model and Data URLs
# =============================
MODEL_URL = "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/best_model.pkl"
DATA_URL = "https://drive.google.com/uc?export=download&id=1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"

# =============================
# 🔹 Load Model from GitHub
# =============================
try:
    st.subheader("🧠 Loading Model...")
    response = requests.get(MODEL_URL, timeout=20)
    response.raise_for_status()
    model = pickle.loads(response.content)
    model_name = model.__class__.__name__
    st.success(f"✅ Model Loaded Successfully: **{model_name}**")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# =============================
# 🔹 Load Dataset from Google Drive
# =============================
try:
    st.subheader("📥 Loading Dataset...")
    data_response = requests.get(DATA_URL, allow_redirects=True, timeout=25)
    data_response.raise_for_status()
    df = pd.read_csv(io.StringIO(data_response.text))
    st.success("✅ Dataset Loaded Successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# =============================
# 🔹 Feature and Target Setup
# =============================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("❌ Dataset must have at least two numeric columns.")
    st.stop()

target = st.selectbox("🎯 Select Target Variable", numeric_cols, index=len(numeric_cols)-1)

try:
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
        for f in features:
            if f not in df.columns:
                df[f] = 0
        X = df[features]
    else:
        X = df.drop(columns=[target])
except Exception as e:
    st.error(f"Feature alignment failed: {e}")
    st.stop()

# =============================
# 🔹 Prediction and Evaluation
# =============================
try:
    st.subheader("📈 Model Evaluation")
    y_pred = model.predict(X)
    if target in df.columns:
        y = df[target]
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**Accuracy:** {r2 * 100:.2f}%")

        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, color="blue", alpha=0.6, label="Predicted")
        ax.plot(y, y, color="red", label="Actual")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted ({model_name})")
        ax.legend()
        st.pyplot(fig)
    else:
        st.dataframe(pd.DataFrame({"Prediction": y_pred}))
except Exception as e:
    st.error(f"Prediction failed: {e}")

# =============================
# 🔹 Manual Input Prediction
# =============================
st.markdown("---")
st.subheader("🧮 Try Your Own Input")

try:
    if hasattr(model, "feature_names_in_"):
        input_features = list(model.feature_names_in_)
    else:
        input_features = [c for c in df.columns if c != target]

    user_input = {}
    cols = st.columns(2)
    for i, col_name in enumerate(input_features):
        with cols[i % 2]:
            try:
                default_val = float(df[col_name].mean())
            except Exception:
                default_val = 0.0
            user_input[col_name] = st.number_input(f"{col_name}", value=default_val)

    if st.button("🔮 Predict"):
        input_df = pd.DataFrame([user_input])
        pred_value = model.predict(input_df)[0]
        st.success(f"📈 Predicted {target}: {pred_value:.2f}")
        st.info(f"🧠 Model Used: **{model_name}**")

except Exception as e:
    st.error(f"Manual input prediction failed: {e}")

# =============================
# 🔹 Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Using Best Model</p>
    </div>
    """,
    unsafe_allow_html=True
)
