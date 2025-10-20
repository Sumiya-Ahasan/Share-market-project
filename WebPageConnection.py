import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- App Title ---
st.title("📈 Auto Model Selector - Share Market ML App")

# Sidebar
st.sidebar.header("Upload CSV Data or Use Sample")
use_example = st.sidebar.checkbox("Use example dataset")

# Load data
if use_example:
    df = sns.load_dataset('iris').dropna()
    st.success("Loaded sample dataset: 'iris'")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file or use the example dataset")
        st.stop()

# Show dataset
st.subheader("📊 Dataset Preview")
st.write(df.head())
st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# Numeric columns check
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns for regression.")
    st.stop()

# Target & features
target = st.selectbox("Select Target Variable", numeric_cols)
features = [col for col in numeric_cols if col != target]

# Prepare data
df = df[features + [target]].dropna()
X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- Train Models ---
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

results = {}

st.subheader("⚙️ Training Models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = {"model": model, "r2": r2, "mse": mse}
    st.write(f"{name}: R² = {r2:.4f}, MSE = {mse:.4f}")

# --- Find Best Model ---
best_model_name = max(results, key=lambda x: results[x]["r2"])
best_model = results[best_model_name]["model"]
best_r2 = results[best_model_name]["r2"]
best_mse = results[best_model_name]["mse"]

st.success(f"🏆 Best Model: **{best_model_name}** (R² = {best_r2:.4f})")

# --- Plot Actual vs Predicted ---
st.subheader(f"📉 Actual vs Predicted ({best_model_name})")
fig, ax = plt.subplots()
y_pred_best = best_model.predict(X_test)

ax.scatter(y_test, y_test, color='red', label='Actual', alpha=0.6)
ax.scatter(y_test, y_pred_best, color='blue', label='Predicted', alpha=0.6)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.legend()
st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Auto Model Selector using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
