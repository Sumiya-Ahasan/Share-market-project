import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

# App title
st.title("📈 Share Market Prediction App (Using Trained Model)")

st.write("Enter the required input values below to get a prediction:")

# Example feature names (replace these with your actual ones)
feature_names = ["Open", "High", "Low", "Volume"]

# Create input fields dynamically
user_inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature} value:", value=0.0)
    user_inputs.append(value)

# Convert to numpy array
input_array = np.array(user_inputs).reshape(1, -1)

# Predict button
if st.button("🔮 Predict"):
    prediction = model.predict(input_array)[0]
    st.success(f"💹 Predicted Value: **{prediction:.2f}**")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Trained Model Prediction</p>
    </div>
    """,
    unsafe_allow_html=True
)
