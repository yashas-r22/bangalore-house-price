# src/preprocess.py — Data Cleaning & Feature Engineering
# Bangalore House Price Prediction | Enterprise ML Project

import pandas as pd
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data(path: str = "data/Bengaluru_House_Data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 2. BASIC CLEANING
# ─────────────────────────────────────────────────────────────
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop low-signal columns
    df.drop(columns=["society", "balcony"], errors="ignore", inplace=True)

    # Fill missing values
    df["location"].fillna("Sarjapur Road", inplace=True)
    df["size"].fillna("2 BHK", inplace=True)
    df["bath"].fillna(df["bath"].median(), inplace=True)

    # Extract BHK from size string (e.g. "3 BHK" → 3)
    df["bhk"] = df["size"].apply(
        lambda x: int(str(x).split()[0]) if pd.notnull(x) else 2
    )

    log.info(f"After basic clean: {df.shape[0]} rows")
    return df


# ─────────────────────────────────────────────────────────────
# 3. CONVERT SQFT (handles ranges like "1000-1500")
# ─────────────────────────────────────────────────────────────
def convert_sqft(x):
    try:
        if "-" in str(x):
            parts = str(x).split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except Exception:
        return np.nan


def engineer_sqft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
    df.dropna(subset=["total_sqft", "price"], inplace=True)
    df = df[df["total_sqft"] > 0]
    return df


# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Price per sqft — key derived feature
    df["price_per_sqft"] = df["price"] * 1e5 / df["total_sqft"]

    # Sqft per BHK — detects mislabelled listings
    df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]

    # Availability encoding
    df["is_ready_to_move"] = df["availability"].apply(
        lambda x: 1 if str(x).strip().lower() == "ready to move" else 0
    )

    # Area type encoding
    area_map = {
        "Super built-up  Area": 0,
        "Built-up  Area": 1,
        "Plot  Area": 2,
        "Carpet  Area": 3,
    }
    df["area_type_enc"] = df["area_type"].map(area_map).fillna(0)

    log.info("Feature engineering complete")
    return df


# ─────────────────────────────────────────────────────────────
# 5. OUTLIER REMOVAL
# ─────────────────────────────────────────────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial = len(df)

    # Rule 1: Minimum sqft per BHK
    df = df[df["sqft_per_bhk"] >= 300]

    # Rule 2: Bathrooms sanity check
    df = df[df["bath"] < df["bhk"] + 2]

    # Rule 3: Price per sqft per location (mean ± 1 std)
    clean_parts = []
    for loc, sub in df.groupby("location"):
        mean = sub["price_per_sqft"].mean()
        std = sub["price_per_sqft"].std()
        filtered = sub[
            (sub["price_per_sqft"] >= mean - std)
            & (sub["price_per_sqft"] <= mean + std)
        ]
        clean_parts.append(filtered)

    df = pd.concat(clean_parts, ignore_index=True)
    log.info(f"Outlier removal: {initial} → {len(df)} rows ({initial - len(df)} removed)")
    return df


# ─────────────────────────────────────────────────────────────
# 6. LOCATION ENCODING (group rare locations → "other")
# ─────────────────────────────────────────────────────────────
def encode_locations(df: pd.DataFrame, min_count: int = 10):
    df = df.copy()

    loc_counts = df["location"].value_counts()
    rare = loc_counts[loc_counts < min_count].index
    df["location"] = df["location"].apply(lambda x: "other" if x in rare else x)

    # One-hot encode locations
    location_dummies = pd.get_dummies(df["location"], prefix="loc")
    df = pd.concat([df, location_dummies], axis=1)

    locations = sorted(df["location"].unique().tolist())
    location_cols = location_dummies.columns.tolist()

    log.info(f"Encoded {len(locations)} unique locations")
    return df, locations, location_cols


# ─────────────────────────────────────────────────────────────
# 7. FULL PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline(path: str = "data/Bengaluru_House_Data.csv"):
    df = load_data(path)
    df = basic_clean(df)
    df = engineer_sqft(df)
    df = feature_engineer(df)
    df = remove_outliers(df)
    df, locations, location_cols = encode_locations(df)
    return df, locations, location_cols