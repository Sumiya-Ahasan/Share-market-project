import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import io
import requests
from sklearn.metrics import mean_squared_error, r2_score

# =============================
# --- App Config
# =============================
st.set_page_config(page_title="📊 Share Market Prediction", page_icon="💹", layout="wide")
st.title("📊 Share Market Prediction App (Local Model)")

# =============================
# --- Load Pre-trained Model (from uploaded file)
# =============================
st.subheader("🧠 Loading Trained Model...")
uploaded_model = "best_model.pkl"  # <-- your uploaded model file name

try:
    with open(uploaded_model, "rb") as f:
        model = pickle.load(f)
    model_name = model.__class__.__name__
    st.success(f"✅ Model Loaded Successfully: **{model_name}**")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# =============================
# --- Load Dataset from Google Drive
# =============================
st.subheader("📥 Loading Dataset...")
DATA_URL = "https://drive.google.com/uc?export=download&id=1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"

try:
    response = requests.get(DATA_URL, timeout=25)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna(0)

    st.success("✅ Dataset Loaded Successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# =============================
# --- Target Selection
# =============================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("❌ Dataset must contain at least two numeric columns.")
    st.stop()

target = st.selectbox("🎯 Select Target Variable", numeric_cols, index=len(numeric_cols) - 1)

# =============================
# --- Align Features with Model
# =============================
try:
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.warning(f"⚠️ Missing columns in dataset: {missing}. Filling them with 0.")
            for col in missing:
                df[col] = 0
        X = df[features]
    else:
        st.info("ℹ️ Model has no feature metadata; using all numeric columns.")
        X = df.select_dtypes(include=np.number)
except Exception as e:
    st.error(f"⚠️ Feature alignment failed: {e}")
    st.stop()

# =============================
# --- Prediction
# =============================
st.subheader("📈 Prediction & Evaluation")

try:
    y_pred = model.predict(X)

    if target in df.columns:
        y = df[target].fillna(df[target].mean())
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        acc = r2 * 100

        st.write(f"**Model Used:** {model_name}")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**Approx Accuracy:** {acc:.2f}%")

        # --- Plot ---
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, color="blue", alpha=0.6, label="Predicted")
        ax.plot(y, y, color="red", label="Actual")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Actual vs Predicted ({model_name})")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("ℹ️ Target not found in dataset — showing only predictions.")
        st.dataframe(pd.DataFrame({"Prediction": y_pred}))
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")

# =============================
# --- Manual Input Prediction
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
            val = st.number_input(
                f"{col_name}",
                value=float(df[col_name].mean()) if col_name in df.columns else 0.0
            )
            user_input[col_name] = val

    if st.button("🔮 Predict"):
        input_df = pd.DataFrame([user_input])
        pred_value = model.predict(input_df)[0]
        st.success(f"📈 Predicted {target}: {pred_value:.2f}")
        st.info(f"🧠 Model Used: **{model_name}**")
except Exception as e:
    st.error(f"⚠️ Manual input prediction failed: {e}")

# =============================
# --- Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Using Local Trained Model</p>
    </div>
    """,
    unsafe_allow_html=True
)
