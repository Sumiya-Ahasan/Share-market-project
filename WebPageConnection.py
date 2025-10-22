import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import io
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 Share Market Prediction Dashboard")

# =============================
# 🔹 Model Links
# =============================
MODEL_URLS = {
    "Linear Regression": "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/linear_reg.pkl",
    "XGBoost Regressor": "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/xgb_model.pkl",
    "Support Vector Machine (SVM)": "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/svm_model.pkl",
    "Random Forest": "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/rf_model.pkl",
}

# =============================
# 🔹 Dataset Link
# =============================
DATA_URL = "https://drive.google.com/uc?export=download&id=1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"

# =============================
# 🔹 Model Selection
# =============================
selected_model_name = st.selectbox("🧠 Select Model", list(MODEL_URLS.keys()))
MODEL_URL = MODEL_URLS[selected_model_name]
st.write(f"✅ Selected Model: **{selected_model_name}**")

# =============================
# 🔹 Load Model
# =============================
try:
    response = requests.get(MODEL_URL, timeout=15)
    response.raise_for_status()
    with open("model.pkl", "wb") as f:
        f.write(response.content)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success(f"Model loaded successfully: {selected_model_name}")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# =============================
# 🔹 Load Dataset
# =============================
try:
    data_response = requests.get(DATA_URL, allow_redirects=True, timeout=20)
    data_response.raise_for_status()
    df = pd.read_csv(io.StringIO(data_response.text))
    st.dataframe(df.head())
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# =============================
# 🔹 Target Variable
# =============================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns.")
    st.stop()
target = st.selectbox("🎯 Select Target Variable", numeric_cols, index=len(numeric_cols) - 1)

# =============================
# 🔹 Feature Alignment
# =============================
try:
    if hasattr(model, "feature_names_in_"):
        required_features = list(model.feature_names_in_)
        for col in required_features:
            if col not in df.columns:
                df[col] = 0
        X = df[required_features]
    else:
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
        st.subheader("🏆 Model Performance Summary")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**Accuracy:** {r2 * 100:.2f}%")

        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, color='blue', alpha=0.6, label='Predicted')
        ax.plot(y, y, color='red', label='Actual')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted ({selected_model_name})")
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
st.subheader("🧮 Try Manual Prediction")

try:
    if hasattr(model, "feature_names_in_"):
        input_features = list(model.feature_names_in_)
    else:
        input_features = [c for c in df.columns if c != target]

    user_input = {}
    cols = st.columns(2)
    for i, feature in enumerate(input_features):
        with cols[i % 2]:
            try:
                default = float(df[feature].mean()) if feature in df.columns else 0.0
            except Exception:
                default = 0.0
            user_input[feature] = st.number_input(feature, value=default)

    if st.button("🔮 Predict"):
        input_df = pd.DataFrame([user_input])
        result = model.predict(input_df)[0]
        st.success(f"📈 Predicted {target}: {result:.2f}")
except Exception as e:
    st.error(f"Manual input failed: {e}")

# =============================
# 🔹 Comparison Table
# =============================
st.markdown("---")
st.subheader("📊 Model Performance Comparison")
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "XGBoost Regressor", "Support Vector Machine (SVM)", "Random Forest"],
    "Accuracy (%)": [92.08, 92.45, 93.86, 90.57]
})
st.table(comparison)

# =============================
# 🔹 Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Multi-Model Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)
