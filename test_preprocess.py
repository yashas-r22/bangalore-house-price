# tests/test_preprocess.py — Unit Tests
# Run with: pytest tests/

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from src.preprocess import (
    basic_clean,
    convert_sqft,
    engineer_sqft,
    feature_engineer,
    remove_outliers,
)


# ── Fixtures ────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "area_type":    ["Super built-up  Area", "Built-up  Area"],
        "availability": ["Ready to Move", "Dec 2025"],
        "location":     ["Whitefield", "Koramangala"],
        "size":         ["2 BHK", "3 BHK"],
        "society":      ["Society A", "Society B"],
        "total_sqft":   ["1200", "1000-1500"],
        "bath":         [2.0, None],
        "balcony":      [1, 2],
        "price":        [80.0, 130.0],
    })


# ── Tests ────────────────────────────────────────────────────
def test_basic_clean_extracts_bhk(sample_df):
    df = basic_clean(sample_df)
    assert "bhk" in df.columns
    assert df["bhk"].iloc[0] == 2
    assert df["bhk"].iloc[1] == 3


def test_basic_clean_fills_bath(sample_df):
    df = basic_clean(sample_df)
    assert df["bath"].isnull().sum() == 0


def test_convert_sqft_range():
    assert convert_sqft("1000-1500") == 1250.0


def test_convert_sqft_plain():
    assert convert_sqft("1200") == 1200.0


def test_convert_sqft_invalid():
    assert np.isnan(convert_sqft("abc sqft"))


def test_feature_engineer_adds_columns(sample_df):
    df = basic_clean(sample_df)
    df = engineer_sqft(df)
    df = feature_engineer(df)
    assert "price_per_sqft" in df.columns
    assert "sqft_per_bhk" in df.columns
    assert "is_ready_to_move" in df.columns


def test_is_ready_to_move_encoding(sample_df):
    df = basic_clean(sample_df)
    df = engineer_sqft(df)
    df = feature_engineer(df)
    assert df["is_ready_to_move"].iloc[0] == 1   # "Ready to Move"
    assert df["is_ready_to_move"].iloc[1] == 0   # "Dec 2025"