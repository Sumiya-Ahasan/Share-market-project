import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- App Title ---
st.title("📈 Share Market Best Model Predictor")
st.write("Enter past closing prices — the app will test multiple models and pick the best one for prediction!")

# --- User Input Section ---
st.subheader("💰 Enter Previous Closing Prices")

num_days = st.number_input("How many previous days' closing prices do you want to enter?", min_value=2, max_value=30, value=5, step=1)

st.write(f"Please enter the closing prices for the last {num_days} days:")
prices = []
for i in range(int(num_days)):
    value = st.number_input(f"Close Price - Day {i+1}", value=0.0, format="%.2f")
    prices.append(value)

# --- Prediction Button ---
if st.button("🔮 Predict Next Day Price"):
    try:
        # Convert to numpy array
        prices = np.array(prices)

        # Create simple dataset for supervised learning (predict next from previous values)
        X = []
        y = []
        for i in range(len(prices) - 1):
            X.append(prices[:i+1])
            y.append(prices[i+1])

        # Pad shorter sequences to same length
        max_len = max(len(x) for x in X)
        X_padded = np.array([np.pad(x, (0, max_len - len(x)), 'constant', constant_values=0) for x in X])
        y = np.array(y)

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_padded)

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Define models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
        }

        # Train and evaluate models
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            results[name] = {"model": model, "r2": r2, "mse": mse}

        # Find best model
        best_model_name = max(results, key=lambda x: results[x]["r2"])
        best_model = results[best_model_name]["model"]
        best_r2 = results[best_model_name]["r2"]
        best_mse = results[best_model_name]["mse"]

        # Prepare input for next-day prediction
        user_input = np.array(prices).reshape(1, -1)
        if user_input.shape[1] < X_padded.shape[1]:
            user_input = np.pad(user_input, ((0, 0), (0, X_padded.shape[1] - user_input.shape[1])), 'constant', constant_values=0)
        user_input_scaled = scaler.transform(user_input)
        next_day_price = best_model.predict(user_input_scaled)[0]

        # Display best model and prediction
        st.success(f"🏆 Best Model: **{best_model_name}** (R² = {best_r2:.3f})")
        st.write(f"💹 Predicted Next Day Close Price: **{next_day_price:.2f}**")

        # --- Plot ---
        st.subheader("📊 Price Trend Visualization")
        fig, ax = plt.subplots()
        days = [f"Day {i+1}" for i in range(len(prices))] + ["Predicted Next Day"]
        all_prices = list(prices) + [next_day_price]

        ax.plot(days, all_prices, marker='o', linestyle='-', color='blue', label='Price Trend')
        ax.scatter(days[-1], all_prices[-1], color='red', label='Predicted Price', s=100)
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.set_title(f"Actual Prices & Predicted Next Price ({best_model_name})")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Share Market ML App | Auto Model Selection</p>
    </div>
    """,
    unsafe_allow_html=True
)
