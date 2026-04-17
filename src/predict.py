# src/predict.py — Prediction Utilities
# Bangalore House Price Prediction | Enterprise ML Project

import pandas as pd
import numpy as np
import pickle
import json
import logging

log = logging.getLogger(__name__)


def load_artifacts(model_dir="models"):
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/locations.pkl", "rb") as f:
        locations = pickle.load(f)
    with open(f"{model_dir}/location_cols.pkl", "rb") as f:
        location_cols = pickle.load(f)
    with open(f"{model_dir}/features.pkl", "rb") as f:
        features = pickle.load(f)
    with open(f"{model_dir}/model_report.json", "r") as f:
        report = json.load(f)
    return model, locations, location_cols, features, report


def predict_price(
    model,
    location: str,
    total_sqft: float,
    bhk: int,
    location_cols: list,
    features: list,
    is_ready_to_move: int = 1,
    area_type_enc: int = 0,
) -> dict:
    """
    Returns predicted price in Lakhs with confidence range.
    """
    sqft_per_bhk = total_sqft / bhk

    # Build base row
    row = {
        "total_sqft": total_sqft,
        "bhk": bhk,
        "sqft_per_bhk": sqft_per_bhk,
        "is_ready_to_move": is_ready_to_move,
        "area_type_enc": area_type_enc,
    }

    # One-hot location
    loc_key = f"loc_{location}"
    for col in location_cols:
        row[col] = 1 if col == loc_key else 0

    # Build dataframe aligned to training features
    input_df = pd.DataFrame([row])
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]

    price = model.predict(input_df)[0]
    price_per_sqft = price * 1e5 / total_sqft

    return {
        "price_lakhs": round(price, 2),
        "price_rupees": int(price * 1e5),
        "price_per_sqft": round(price_per_sqft, 0),
        "price_low": round(price * 0.90, 2),   # ±10% confidence band
        "price_high": round(price * 1.10, 2),
        "carpet_area_sqft": int(total_sqft * 0.70),
        "emi_20yr_lakhs": round(_emi(price * 0.80 * 1e5, 20, 8.5) / 1e5, 2),
    }


def _emi(principal, years, annual_rate):
    r = annual_rate / 12 / 100
    n = years * 12
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)