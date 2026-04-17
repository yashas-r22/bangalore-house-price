# src/train.py — Model Training, Comparison & Saving
# Bangalore House Price Prediction | Enterprise ML Project

import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from src.preprocess import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# FEATURE COLUMNS (NO BATHROOMS — see README for why)
# ─────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "total_sqft",
    "bhk",
    "sqft_per_bhk",
    "is_ready_to_move",
    "area_type_enc",
]


def get_all_features(location_cols):
    return BASE_FEATURES + location_cols


# ─────────────────────────────────────────────────────────────
# METRICS HELPER
# ─────────────────────────────────────────────────────────────
def compute_metrics(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 5-fold CV on training data
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    return {
        "MAE (Lakhs)": round(mae, 4),
        "RMSE (Lakhs)": round(rmse, 4),
        "R² (Test)": round(r2, 4),
        "CV R² (mean)": round(cv.mean(), 4),
        "CV R² (std)": round(cv.std(), 4),
    }


# ─────────────────────────────────────────────────────────────
# TRAIN ALL MODELS
# ─────────────────────────────────────────────────────────────
def train_all(data_path="data/Bengaluru_House_Data.csv"):
    log.info("=== Starting Training Pipeline ===")

    # 1. Preprocess
    df, locations, location_cols = run_pipeline(data_path)
    features = get_all_features(location_cols)

    # Fill any missing location dummy columns with 0
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features]
    y = df["price"]

    log.info(f"Features: {len(features)} | Samples: {len(X)}")

    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }

    # 4. Train & compare
    results = {}
    trained_models = {}

    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'CV R²':>12}")
    print("=" * 70)

    for name, model in models.items():
        log.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        trained_models[name] = model

        print(
            f"{name:<25} "
            f"{metrics['MAE (Lakhs)']:>10.2f} "
            f"{metrics['RMSE (Lakhs)']:>10.2f} "
            f"{metrics['R² (Test)']:>10.4f} "
            f"{metrics['CV R² (mean)']:>10.4f} ± {metrics['CV R² (std)']:.4f}"
        )

    print("=" * 70)

    # 5. Pick best model by CV R²
    best_name = max(results, key=lambda k: results[k]["CV R² (mean)"])
    best_model = trained_models[best_name]
    log.info(f"\n✅ Best Model: {best_name} | CV R²: {results[best_name]['CV R² (mean)']:.4f}")

    # 6. Feature importance (if tree-based)
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        fi = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        feature_importance = {k: round(float(v), 6) for k, v in fi[:20]}
        print("\nTop 10 Feature Importances:")
        for feat, imp in list(feature_importance.items())[:10]:
            print(f"  {feat:<35} {imp:.4f}")

    # 7. Save artifacts
    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("models/locations.pkl", "wb") as f:
        pickle.dump(locations, f)

    with open("models/location_cols.pkl", "wb") as f:
        pickle.dump(location_cols, f)

    with open("models/features.pkl", "wb") as f:
        pickle.dump(features, f)

    # Save model report
    report = {
        "trained_at": datetime.now().isoformat(),
        "best_model": best_name,
        "best_metrics": results[best_name],
        "all_results": results,
        "feature_importance_top20": feature_importance,
        "num_features": len(features),
        "num_locations": len(locations),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    with open("models/model_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log.info("✅ All artifacts saved to models/")
    return best_model, locations, location_cols, features, results


if __name__ == "__main__":
    train_all()