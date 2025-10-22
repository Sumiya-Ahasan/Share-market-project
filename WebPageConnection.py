import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import io

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

st.set_page_config(page_title="Smart Share Market Prediction", page_icon="📊", layout="wide")
st.title("🤖 Smart Share Market Prediction (with ML Pipelines)")

# =============================
# 🔹 Dataset Link
# =============================
DATA_URL = "https://drive.google.com/uc?export=download&id=1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"

# =============================
# 🔹 Load Dataset
# =============================
try:
    st.subheader("📥 Loading Dataset...")
    response = requests.get(DATA_URL, allow_redirects=True, timeout=25)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    st.success("✅ Dataset Loaded Successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# =============================
# 🔹 Select Target Column
# =============================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("❌ Dataset must have at least two numeric columns.")
    st.stop()

target = st.selectbox("🎯 Select Target Variable", numeric_cols, index=len(numeric_cols) - 1)
X = df.drop(columns=[target])
y = df[target]

# =============================
# 🔹 Define Pipelines for Each Model
# =============================
pipelines = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "XGBoost Regressor": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", XGBRegressor(random_state=42, n_estimators=200))
    ]),
    "Support Vector Machine (SVM)": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(random_state=42, n_estimators=150))
    ])
}

# =============================
# 🔹 Train & Evaluate All Models
# =============================
st.subheader("🧠 Training and Evaluating Models...")

performance = {}
for name, pipeline in pipelines.items():
    try:
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        performance[name] = {"model": pipeline, "r2": r2, "mse": mse}
        st.write(f"✅ {name}: R² = {r2:.4f}, MSE = {mse:.2f}")
    except Exception as e:
        st.warning(f"⚠️ {name} failed: {e}")

# =============================
# 🔹 Auto Select Best Model
# =============================
best_model_name = max(performance, key=lambda k: performance[k]["r2"])
best_model = performance[best_model_name]["model"]
best_r2 = performance[best_model_name]["r2"]

st.success(f"🏆 Best Model Selected Automatically: **{best_model_name}** (R² = {best_r2:.4f})")

# =============================
# 🔹 Plot Actual vs Predicted
# =============================
try:
    y_pred = best_model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.6, color="blue", label="Predicted")
    ax.plot(y, y, color="red", label="Actual")
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
st.subheader("🧮 Try Your Own Input (Auto Pipeline Applied)")

user_input = {}
cols = st.columns(2)
for i, col_name in enumerate(X.columns):
    with cols[i % 2]:
        try:
            default_val = float(df[col_name].mean())
        except Exception:
            default_val = 0.0
        user_input[col_name] = st.number_input(col_name, value=default_val)

if st.button("🔮 Predict Automatically"):
    input_df = pd.DataFrame([user_input])
    prediction = best_model.predict(input_df)[0]
    st.success(f"📈 Predicted {target}: {prediction:.2f}")
    st.info(f"🤖 Model Used: **{best_model_name}** (R² = {best_r2:.4f})")

# =============================
# 🔹 Model Comparison Table
# =============================
st.markdown("---")
st.subheader("📊 Model Performance Comparison")

perf_df = pd.DataFrame({
    "Model": performance.keys(),
    "R² Score": [round(v["r2"], 4) for v in performance.values()],
    "MSE": [round(v["mse"], 2) for v in performance.values()]
}).sort_values(by="R² Score", ascending=False)

st.table(perf_df)

# =============================
# 🔹 Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Smart Share Market ML App | Pipeline-Integrated Models</p>
    </div>
    """,
    unsafe_allow_html=True
)
