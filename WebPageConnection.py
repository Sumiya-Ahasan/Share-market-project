import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# =====================================================
# 🔹 1. Load dataset (Google Drive or local CSV)
# =====================================================
DATA_URL = "https://drive.google.com/uc?export=download&id=1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"

print("📥 Loading dataset from Google Drive...")
df = pd.read_csv(DATA_URL)
print("✅ Dataset loaded successfully!")

# =====================================================
# 🔹 2. Basic preprocessing / feature engineering
# =====================================================
# উদাহরণ: ধরো তুমি share market ডেটা নিয়ে কাজ করছো
# আমরা কিছু নতুন feature তৈরি করব যাতে model better হয়

if "Close" in df.columns:
    df["Close_lag_1"] = df["Close"].shift(1)
    df["Return_1d"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["Volatility_5"] = df["Close"].rolling(window=5).std()

# Missing value fill
df = df.fillna(0)

# =====================================================
# 🔹 3. Select features and target
# =====================================================
target = "Close"  # Change if needed
feature_cols = [c for c in df.columns if c != target and df[c].dtype != "O"]

X = df[feature_cols]
y = df[target]

print(f"✅ Selected {len(feature_cols)} features: {feature_cols}")

# =====================================================
# 🔹 4. Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================
# 🔹 5. Build pipeline (Scaler + RandomForest)
# =====================================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

print("🚀 Training model...")
pipeline.fit(X_train, y_train)

# =====================================================
# 🔹 6. Evaluate
# =====================================================
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"📊 Model Evaluation -> R²: {r2:.4f}, MSE: {mse:.2f}")

# =====================================================
# 🔹 7. Save model (pipeline includes preprocessing)
# =====================================================
with open("best_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model saved successfully as best_model.pkl")
print("🧠 This model includes feature scaling + feature names in metadata.")
