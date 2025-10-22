import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import io
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Smart Share Market Prediction", page_icon="📊", layout="wide")
st.title("🤖 Smart Share Market Prediction (Auto Model Selector)")

# =============================
# 🔹 Model URLs (GitHub)
# =============================
MODEL_URLS = {
    "Linear Regression": "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/linear_reg.pkl",
    "XGBoost Regressor": "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/xgb_model.pkl",
    "Support Vector Machine (SVM)": "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/svm_model.pkl",
    "Random Forest": "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/rf_model.pkl"
}

# =============================
# 🔹 Dataset Link (Google Drive)
# =============================
DATA_URL = "https://drive.google.com/uc?export=download&id=1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"

# =============================
# 🔹 Load Dataset
# =============================
st.header("📥 Load Dataset")
try:
    data_response = requests.get(DATA_URL, allow_redirects=True, timeout=25)
    data_response.raise_for_status()
    df = pd.read_csv(io.StringIO(data_response.text))
    st.success("✅ Dataset Loaded Successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# =============================
# 🔹 Identify Numeric Columns
# =============================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("❌ Dataset must have at least two numeric columns.")
    st.stop()

target = st.selectbox("🎯 Select Target Variable", numeric_cols, index=len(numeric_cols) - 1)
X = df.drop(columns=[target])
y = df[target]

# =============================
# 🔹 Helper: Load Model
# =============================
@st.cache_resource
def load_model_from_url(url):
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    with open("temp_model.pkl", "wb") as f:
        f.write(response.content)
    with open("temp_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# =============================
# 🔹 Evaluate All Models
# =============================
st.header("🧠 Evaluating Models...")
model_performance = {}

for name, url in MODEL_URLS.items():
    try:
        model = load_model_from_url(url)
        # Feature alignment
        if hasattr(model, "feature_names_in_"):
            needed_features = list(model.feature_names_in_)
            for col in needed_features:
                if col not in X.columns:
                    X[col] = 0
            X_model = X[needed_features]
        else:
            X_model = X.select_dtypes(include=np.number)

        y_pred = model.predict(X_model)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        model_performance[name] = {"r2": r2, "mse": mse, "model": model}
        st.write(f"✅ {name}: R² = {r2:.4f}, MSE = {mse:.2f}")
    except Exception as e:
        st.warning(f"⚠️ {name} failed: {e}")

# =============================
# 🔹 Auto-select Best Model
# =============================
if not model_performance:
    st.error("❌ No model could be evaluated successfully.")
    st.stop()

best_model_name = max(model_performance, key=lambda m: model_performance[m]["r2"])
best_model = model_performance[best_model_name]["model"]
best_r2 = model_performance[best_model_name]["r2"]
st.success(f"🏆 Best Model Selected Automatically: **{best_model_name}** (R² = {best_r2:.4f})")

# =============================
# 🔹 Visualization
# =============================
try:
    y_pred_best = best_model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred_best, color='blue', alpha=0.6, label='Predicted')
    ax.plot(y, y, color='red', label='Actual')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Actual vs Predicted ({best_model_name})")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Plotting failed: {e}")

# =============================
# 🔹 Manual Input Prediction
# =============================
st.markdown("---")
st.header("🧮 Manual Prediction (Auto Model)")

input_features = [c for c in df.columns if c != target]
user_input = {}
cols = st.columns(2)

for i, col_name in enumerate(input_features):
    with cols[i % 2]:
        try:
            default_val = float(df[col_name].mean())
        except Exception:
            default_val = 0.0
        user_input[col_name] = st.number_input(col_name, value=default_val)

if st.button("🔮 Predict Automatically"):
    input_df = pd.DataFrame([user_input])
    pred_value = best_model.predict(input_df)[0]
    st.success(f"📈 Predicted {target}: {pred_value:.2f}")
    st.info(f"🤖 Model Used: **{best_model_name}** (R² = {best_r2:.4f})")

# =============================
# 🔹 Model Comparison Table
# =============================
st.markdown("---")
st.header("📊 Model Performance Comparison")

comparison_df = pd.DataFrame({
    "Model": list(model_performance.keys()),
    "R² Score": [round(v["r2"], 4) for v in model_performance.values()],
    "MSE": [round(v["mse"], 2) for v in model_performance.values()]
}).sort_values(by="R² Score", ascending=False)
st.table(comparison_df)

# =============================
# 🔹 Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Smart Share Market ML App | Auto Model Selector</p>
    </div>
    """,
    unsafe_allow_html=True
)
