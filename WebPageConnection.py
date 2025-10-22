import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import io
from sklearn.metrics import mean_squared_error, r2_score

# --- App Title ---
st.title("📊 Share Market Prediction (Auto + Manual Input Mode)")

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
    st.success("✅ Model loaded successfully from GitHub!")
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
    st.success("✅ Dataset loaded successfully from Google Drive!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# =============================
# 🔹 Feature and Target Selection
# =============================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns for regression.")
    st.stop()

target = st.selectbox("🎯 Select Target Variable", numeric_cols, index=len(numeric_cols) - 1)
features = [col for col in numeric_cols if col != target]

X = df[features]
y = df[target]

# =============================
# 🔹 Prediction and Evaluation on Dataset
# =============================
try:
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    accuracy = r2 * 100

    st.subheader("🏆 Model Performance Summary")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**Accuracy:** {accuracy:.2f}%")

    # --- Plot Actual vs Predicted ---
    st.subheader("📉 Actual vs Predicted (Dataset)")
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, color='blue', alpha=0.6, label='Predicted')
    ax.plot(y, y, color='red', label='Actual')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted (Pretrained Model)")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Prediction failed: {e}")

# =============================
# 🔹 Manual Input Prediction Section
# =============================
st.markdown("---")
st.subheader("🧮 Try Your Own Input")

try:
    # Model feature names (pipeline stores these automatically)
    if hasattr(model, "feature_names_in_"):
        input_features = list(model.feature_names_in_)
    else:
        input_features = features

    user_input = {}
    st.info("Enter values for each input feature below:")
    cols = st.columns(2)

    for i, col_name in enumerate(input_features):
        with cols[i % 2]:
            value = st.number_input(f"{col_name}", 
                                    value=float(df[col_name].mean()) if col_name in df.columns else 0.0)
            user_input[col_name] = value

    if st.button("🔮 Predict from Input"):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        st.success(f"📈 **Predicted {target}: {pred:.2f}**")

except Exception as e:
    st.error(f"⚠️ Manual input prediction failed: {e}")

# =============================
# 🔹 Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Auto Data + Manual Input Mode</p>
    </div>
    """,
    unsafe_allow_html=True
)
