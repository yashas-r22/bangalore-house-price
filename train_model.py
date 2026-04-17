# train_model.py — Run this FIRST to train and save the model
# Bangalore House Price Prediction | Enterprise ML Project
#
# Usage:
#   python train_model.py
#   python train_model.py --data data/Bengaluru_House_Data.csv
#
# This script:
#   1. Loads & cleans the raw dataset
#   2. Runs full EDA and saves charts to assets/
#   3. Trains 6 ML models and compares them
#   4. Saves the best model + metadata to models/

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_data, basic_clean, engineer_sqft, feature_engineer
from src.eda import run_eda
from src.train import train_all


def main():
    parser = argparse.ArgumentParser(description="Train Bangalore House Price Model")
    parser.add_argument("--data", default="data/Bengaluru_House_Data.csv", help="Path to CSV dataset")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA charts generation")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"\n❌ Dataset not found at: {args.data}")
        print("   Please download 'Bengaluru_House_Data.csv' from Kaggle and place it in data/")
        print("   URL: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data\n")
        sys.exit(1)

    print("\n🏠 Bangalore House Price Predictor — Training Pipeline")
    print("=" * 55)

    # Optional EDA
    if not args.skip_eda:
        print("\n📊 Running EDA...")
        df_eda = load_data(args.data)
        df_eda = basic_clean(df_eda)
        df_eda = engineer_sqft(df_eda)
        df_eda = feature_engineer(df_eda)
        run_eda(df_eda)

    # Train
    print("\n🤖 Training Models...")
    best_model, locations, location_cols, features, results = train_all(args.data)

    print("\n✅ Training complete!")
    print("   Artifacts saved to models/")
    print("   EDA charts saved to assets/")
    print("\n🚀 Next step: run the Streamlit app")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()