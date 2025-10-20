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
st.title("📈 Share Market Prediction App")

# --- Load Dataset from Google Drive ---
FILE_ID = "1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"  # your file ID
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

try:
    st.info("📦 Loading dataset from Google Drive...")
    response = requests.get(URL)
    response.raise_for_status()  # check for errors
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

# --- Model Selection Sidebar ---
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select a Regression Model",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

# --- Initialize Models ---
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=200, random_state=42)
elif model_choice == "XGBoost":
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )

# --- Train Model ---
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluate Model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

# --- Display Results ---
st.subheader(f"📊 Model Evaluation: {model_choice}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**Model Accuracy:** {accuracy:.2f}%")

# --- Plot ---
st.subheader("📉 Actual vs Predicted")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', label='Predicted')
ax.plot(y_test, y_test, color='red', label='Actual')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title(f"Actual vs Predicted ({model_choice})")
ax.legend()
st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Powered by Streamlit & XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True
)
