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

# App title
st.title("📈  Hello! Share Market")

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

# Model feature selection
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns for regression.")
    st.stop()

# Target and feature selection
target = st.selectbox("Select target variable", numeric_cols)
features = st.multiselect(
    "Select input feature columns",
    [col for col in numeric_cols if col != target],
    default=[col for col in numeric_cols if col != target]
)

if len(features) == 0:
    st.error("Please select at least one feature")
    st.stop()

# Prepare data
df = df[features + [target]].dropna()
X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Model selection ---
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select a Regression Model",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

# --- Model initialization ---
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

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

# --- Display Results ---
st.subheader(f"📊 Model Evaluation: {model_choice}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")
st.write(f"Model Accuracy: {accuracy:.2f}%")

# --- Plot ---
st.subheader("📉 Actual vs Predicted")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title(f"Actual vs Predicted ({model_choice})")
st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <p>Developed with  by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Powered by Streamlit & XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True
)
