import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- App Title ---
st.title("📈 Share Market Prediction App (Auto Best Model)")

# --- Load Dataset from Google Drive ---
FILE_ID = "1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"  # Your file ID
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

try:
    st.info("📦 Loading dataset from Google Drive...")
    response = requests.get(URL)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    st.success("✅ Dataset loaded successfully from Google Drive!")
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# --- Dataset Info ---
st.subheader("📊 Dataset Preview")
st.write(df.head())
st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# --- Model Feature Selection ---
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns for regression.")
    st.stop()

# --- Target and Feature Selection ---
target = st.selectbox("🎯 Select Target Variable", numeric_cols)
features = st.multiselect(
    "🧮 Select Input Feature Columns",
    [col for col in numeric_cols if col != target],
    default=[col for col in numeric_cols if col != target]
)

if len(features) == 0:
    st.error("Please select at least one input feature.")
    st.stop()

# --- Prepare Data ---
df = df[features + [target]].dropna()
X = df[features]
y = df[target]

# --- Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )
}

# --- Train and Evaluate Models ---
results = {}
st.info("🚀 Training models comparing performance...")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"model": model, "mse": mse, "r2": r2}

# --- Find Best Model ---
best_model_name = max(results, key=lambda x: results[x]["r2"])
best_model = results[best_model_name]["model"]
best_mse = results[best_model_name]["mse"]
best_r2 = results[best_model_name]["r2"]
best_accuracy = best_r2 * 100

# --- Display Best Model ---
st.success(f"🏆 Best Model: **{best_model_name}**")
st.write(f"**R² Score:** {best_r2:.4f}")
st.write(f"**Mean Squared Error:** {best_mse:.2f}")
st.write(f"**Model Accuracy:** {best_accuracy:.2f}%")

# --- Plot ---
st.subheader("📉 Actual vs Predicted (Best Model)")
y_pred_best = best_model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_best, color='blue', label='Predicted')
ax.plot(y_test, y_test, color='red', label='Actual')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title(f"Actual vs Predicted ({best_model_name})")
ax.legend()
st.pyplot(fig)

# --- Comparison Table ---
st.subheader("📋 Model Comparison Summary")
comparison_df = pd.DataFrame({
    "Model": list(results.keys()),
    "R² Score": [v["r2"] for v in results.values()],
    "MSE": [v["mse"] for v in results.values()]
})
st.dataframe(comparison_df.style.format({"R² Score": "{:.4f}", "MSE": "{:.2f}"}))

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Auto Model Selection | Powered by Streamlit & XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True
)
