import streamlit as st
import numpy as np
import joblib
import os
import urllib.request

# --- App Title ---
st.title("📈 Share Market Prediction App")
st.write("Predict the next day's closing price using your trained model.")

# --- Model File from GitHub ---
MODEL_URL = "https://raw.githubusercontent.com/Sumiya-Ahasan/Share-market-project/main/best_model.pkl"
MODEL_PATH = "best_model.pkl"

# --- Download Model if not Found ---
if not os.path.exists(MODEL_PATH):
    try:
        st.info("📦 Downloading trained model from GitHub...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to download model: {e}")
        st.stop()

# --- Load Model ---
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- User Input Section ---
st.subheader("🧮 Enter Closing Prices")

# Let the user choose how many days of closing prices they want to input
num_days = st.number_input("How many previous days' closing prices?", min_value=1, max_value=30, value=6, step=1)

# Dynamically create input boxes
st.write(f"Enter the closing prices for the last {num_days} days:")
user_inputs = []
for i in range(int(num_days)):
    value = st.number_input(f"Close Price - Day {i+1}", value=0.0, format="%.2f")
    user_inputs.append(value)

# Convert to NumPy array
features = np.array(user_inputs).reshape(1, -1)

# --- Prediction ---
if st.button("🔮 Predict Next Day Close Price"):
    try:
        # Try predicting directly
        prediction = model.predict(features)[0]
        st.success(f"💹 Predicted Next Day Close Price: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")
        st.info("💡 Tip: Make sure the number of inputs matches the number of features the model was trained on.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Powered by Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
