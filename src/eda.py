# src/eda.py — Exploratory Data Analysis
# Bangalore House Price Prediction | Enterprise ML Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("assets", exist_ok=True)


def run_eda(df: pd.DataFrame):
    """Run full EDA and save charts to assets/ folder."""

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Stats:\n{df[['total_sqft','price','bhk','bath']].describe().round(2)}")

    # ── 1. Price distribution ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["price"].hist(bins=60, ax=axes[0], color="#378ADD", edgecolor="white")
    axes[0].set_title("Price Distribution (Lakhs)")
    axes[0].set_xlabel("Price (Lakhs)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}L"))

    np.log1p(df["price"]).hist(bins=60, ax=axes[1], color="#1D9E75", edgecolor="white")
    axes[1].set_title("Log Price Distribution")
    axes[1].set_xlabel("log(Price)")
    plt.tight_layout()
    plt.savefig("assets/price_distribution.png", dpi=150)
    plt.close()
    print("Saved: assets/price_distribution.png")

    # ── 2. Top 15 locations by median price ─────────────────
    top_locs = (
        df.groupby("location")["price"]
        .median()
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    top_locs.plot(kind="barh", ax=ax, color="#7F77DD")
    ax.set_title("Top 15 Locations by Median Price")
    ax.set_xlabel("Median Price (Lakhs)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}L"))
    plt.tight_layout()
    plt.savefig("assets/top_locations.png", dpi=150)
    plt.close()
    print("Saved: assets/top_locations.png")

    # ── 3. BHK vs Price boxplot ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df["bhk"] <= 6].boxplot(
        column="price", by="bhk", ax=ax,
        patch_artist=True,
        boxprops=dict(facecolor="#B5D4F4"),
        medianprops=dict(color="#185FA5", linewidth=2),
    )
    ax.set_title("Price Distribution by BHK")
    ax.set_xlabel("BHK")
    ax.set_ylabel("Price (Lakhs)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("assets/bhk_vs_price.png", dpi=150)
    plt.close()
    print("Saved: assets/bhk_vs_price.png")

    # ── 4. Sqft vs Price scatter ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    sample = df[df["price"] < 500].sample(min(2000, len(df)), random_state=42)
    ax.scatter(sample["total_sqft"], sample["price"], alpha=0.3, s=10, color="#D85A30")
    ax.set_title("Total Sqft vs Price")
    ax.set_xlabel("Total Sqft")
    ax.set_ylabel("Price (Lakhs)")
    plt.tight_layout()
    plt.savefig("assets/sqft_vs_price.png", dpi=150)
    plt.close()
    print("Saved: assets/sqft_vs_price.png")

    # ── 5. Correlation heatmap ───────────────────────────────
    numeric_cols = ["total_sqft", "bhk", "bath", "price", "price_per_sqft", "sqft_per_bhk"]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("assets/correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: assets/correlation_heatmap.png")

    # ── 6. Price per sqft by area type ───────────────────────
    if "area_type" in df.columns and "price_per_sqft" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df.groupby("area_type")["price_per_sqft"].median().sort_values().plot(
            kind="barh", ax=ax, color="#EF9F27"
        )
        ax.set_title("Median Price/Sqft by Area Type")
        ax.set_xlabel("Price per Sqft (₹)")
        plt.tight_layout()
        plt.savefig("assets/area_type_price.png", dpi=150)
        plt.close()
        print("Saved: assets/area_type_price.png")

    print("\n✅ EDA complete — all charts saved to assets/")


if __name__ == "__main__":
    from src.preprocess import load_data, basic_clean, engineer_sqft, feature_engineer
    df = load_data()
    df = basic_clean(df)
    df = engineer_sqft(df)
    df = feature_engineer(df)
    run_eda(df)