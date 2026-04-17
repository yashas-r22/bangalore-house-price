# model.py — Bangalore House Price Prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("Bengaluru_House_Data.csv")
print("Shape:", df.shape)
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# ─────────────────────────────────────────
# 2. DROP UNNECESSARY COLUMNS
# ─────────────────────────────────────────
df.drop(columns=["area_type", "society", "balcony", "availability"], inplace=True)

# ─────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────
df["location"].fillna("Sarjapur Road", inplace=True)
df["size"].fillna("2 BHK", inplace=True)
df["bath"].fillna(df["bath"].median(), inplace=True)

# ─────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────

# Extract BHK number from 'size' column (e.g., "2 BHK" → 2)
df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]) if isinstance(x, str) else 0)

# Convert 'total_sqft' to a single float number
def convert_sqft(x):
    try:
        if "-" in str(x):                        # range like "1000-1200" → average
            parts = x.split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df.dropna(subset=["total_sqft"], inplace=True)

# Price per sqft column (useful for outlier removal)
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

# ─────────────────────────────────────────
# 5. REMOVE OUTLIERS
# ─────────────────────────────────────────

# Remove rows where sqft per BHK is less than 300
df = df[df["total_sqft"] / df["bhk"] >= 300]

# Remove price outliers per location using mean & std
def remove_outliers(df):
    df_out = pd.DataFrame()
    for location, sub_df in df.groupby("location"):
        mean = sub_df["price_per_sqft"].mean()
        std  = sub_df["price_per_sqft"].std()
        clean = sub_df[
            (sub_df["price_per_sqft"] > (mean - std)) &
            (sub_df["price_per_sqft"] < (mean + std))
        ]
        df_out = pd.concat([df_out, clean], ignore_index=True)
    return df_out

df = remove_outliers(df)

# Remove properties where bathrooms > bhk + 2
df = df[df["bath"] < df["bhk"] + 2]

print(f"\nShape after cleaning: {df.shape}")

# ─────────────────────────────────────────
# 6. ENCODE LOCATION (Top locations + 'other')
# ─────────────────────────────────────────

# Keep only locations with more than 10 entries, rest → 'other'
location_counts = df["location"].value_counts()
locations_less_10 = location_counts[location_counts <= 10].index
df["location"] = df["location"].apply(
    lambda x: "other" if x in locations_less_10 else x
)

# Label Encode location
le = LabelEncoder()
df["location_encoded"] = le.fit_transform(df["location"])

# Save the location encoder for use in app.py
with open("location_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save unique location list for dropdown in app.py
locations = sorted(df["location"].unique().tolist())
with open("locations.pkl", "wb") as f:
    pickle.dump(locations, f)

print(f"\nUnique locations: {len(locations)}")

# ─────────────────────────────────────────
# 7. DEFINE FEATURES & TARGET
# ─────────────────────────────────────────
features = ["location_encoded", "total_sqft", "bath", "bhk"]
X = df[features]
y = df["price"]  # price in Lakhs

# ─────────────────────────────────────────
# 8. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────
# 9. COMPARE MODELS
# ─────────────────────────────────────────
models = {
    "Linear Regression"      : LinearRegression(),
    "Random Forest"          : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"      : GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("\n📊 Model Comparison:")
print("-" * 45)
best_model = None
best_score = -999

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"{name:<25} MAE: {mae:.2f}  R²: {r2:.4f}")
    if r2 > best_score:
        best_score = r2
        best_model = m
        best_name  = name

print(f"\n✅ Best Model: {best_name} (R² = {best_score:.4f})")

# ─────────────────────────────────────────
# 10. SAVE BEST MODEL
# ─────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ model.pkl saved!")
print("✅ location_encoder.pkl saved!")
print("✅ locations.pkl saved!")