import streamlit as st
import numpy as np
import joblib
import os

# --- App Title ---
st.title("📈 Share Market Prediction App")
st.write("Predict share market outcomes using your pre-trained model!")

# --- Load Model ---
model_path = "best_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model file not found! Please make sure 'best_model.pkl' is in the same directory.")
    st.stop()

# --- Feature Input Section ---
st.subheader("🧮 Enter Feature Values")

# 🔹 Replace these with your actual feature names
feature_names = ["Open", "High", "Low", "Volume"]

# Dynamically create input fields
inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature} value:", value=0.0)
    inputs.append(value)

# Convert inputs to numpy array
input_data = np.array(inputs).reshape(1, -1)

# --- Prediction Section ---
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💹 Predicted Value: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

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
