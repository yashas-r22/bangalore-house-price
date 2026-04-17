# app.py — Enterprise Streamlit Web App
# Bangalore House Price Prediction | Full ML Project

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import sys

# Fix: Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import load_artifacts, predict_price

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.78rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 500;
        color: #111827;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.2rem;
    }
    .price-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        border-radius: 18px;
        padding: 2rem 2.4rem;
        color: white;
        margin: 1.5rem 0;
    }
    .price-hero-label { font-size: 0.85rem; opacity: 0.7; letter-spacing: 0.05em; text-transform: uppercase; }
    .price-hero-value { font-family: 'DM Serif Display', serif; font-size: 3rem; margin: 0.3rem 0; }
    .price-hero-range { font-size: 0.9rem; opacity: 0.75; }
    .insight-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 3px 3px 3px 0;
    }
    .pill-blue  { background: #eff6ff; color: #1d4ed8; }
    .pill-green { background: #f0fdf4; color: #15803d; }
    .pill-amber { background: #fffbeb; color: #92400e; }
    .pill-red   { background: #fef2f2; color: #991b1b; }
    .section-header {
        font-size: 0.75rem;
        font-weight: 500;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.5rem 0 0.8rem;
    }
    .why-box {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: #374151;
        line-height: 1.7;
        margin: 1rem 0;
    }
    .stSelectbox label, .stSlider label, .stRadio label { font-size: 0.85rem !important; color: #374151 !important; }
    div[data-testid="stSidebar"] { background: #f9fafb; }
    .sidebar-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return load_artifacts("models")
    except Exception as e:
        return None, None, None, None, None


import subprocess

model, locations, location_cols, features, report = load_model()

if model is None:
    st.warning("⚙️ Model not found. Training model... please wait ⏳")

    try:
        subprocess.run(["python", "train_model.py"], check=True)

        # Reload after training
        model, locations, location_cols, features, report = load_model()

    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-header">🏠 Property Details</p>', unsafe_allow_html=True)
    st.markdown("Fill in the details to get an instant estimate.")
    st.divider()

    st.markdown('<p class="section-header">Location</p>', unsafe_allow_html=True)
    location = st.selectbox("Locality", options=sorted(locations))

    st.markdown('<p class="section-header">Size & Configuration</p>', unsafe_allow_html=True)
    total_sqft = st.slider("Built-up Area (sqft)", min_value=400, max_value=5000, value=1200, step=50)
    bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5, 6], index=1)

    st.markdown('<p class="section-header">Property Details</p>', unsafe_allow_html=True)
    availability = st.radio("Availability", ["Ready to Move", "Under Construction"], horizontal=True)
    area_type = st.selectbox(
        "Area Type",
        ["Super built-up Area", "Built-up Area", "Carpet Area", "Plot Area"]
    )

    is_ready = 1 if availability == "Ready to Move" else 0
    area_enc = {"Super built-up Area": 0, "Built-up Area": 1, "Carpet Area": 3, "Plot Area": 2}.get(area_type, 0)

    st.divider()
    predict_btn = st.button("🔍 Estimate Price", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">Bangalore House Price Estimator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ML-powered price prediction using Gradient Boosting & Random Forest on real Bangalore listing data</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💰 Price Estimate", "📊 EDA & Insights", "🤖 Model Report"])

# ── TAB 1: PREDICTION ────────────────────────────────────────
with tab1:
    result = predict_price(
        model=model,
        location=location,
        total_sqft=total_sqft,
        bhk=bhk,
        location_cols=location_cols,
        features=features,
        is_ready_to_move=is_ready,
        area_type_enc=area_enc,
    )

    price = result["price_lakhs"]
    price_cr = price / 100

    # Hero price card
    price_str = f"₹ {price:.1f} L" if price < 100 else f"₹ {price_cr:.2f} Cr"
    st.markdown(f"""
    <div class="price-hero">
        <div class="price-hero-label">Estimated Market Price — {location}</div>
        <div class="price-hero-value">{price_str}</div>
        <div class="price-hero-range">
            Range: ₹ {result['price_low']:.1f}L — ₹ {result['price_high']:.1f}L &nbsp;·&nbsp;
            {total_sqft:,} sqft {bhk}BHK &nbsp;·&nbsp; {availability}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Price per sqft</div>
            <div class="metric-value">₹ {result['price_per_sqft']:,.0f}</div>
            <div class="metric-sub">Built-up area rate</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Carpet area ~</div>
            <div class="metric-value">{result['carpet_area_sqft']:,} sqft</div>
            <div class="metric-sub">≈ 70% of built-up</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">EMI estimate</div>
            <div class="metric-value">₹ {result['emi_20yr_lakhs']:.2f}L/mo</div>
            <div class="metric-sub">20yr @ 8.5% · 80% loan</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        sqft_per_bhk = total_sqft // bhk
        quality = "Spacious" if sqft_per_bhk > 700 else ("Comfortable" if sqft_per_bhk > 450 else "Compact")
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Space per BHK</div>
            <div class="metric-value">{sqft_per_bhk} sqft</div>
            <div class="metric-sub">{quality}</div>
        </div>""", unsafe_allow_html=True)

    # Insights
    st.markdown('<p class="section-header">Smart Insights</p>', unsafe_allow_html=True)
    insights = []
    if total_sqft / bhk < 400:
        insights.append(("pill-amber", "⚠️ Less than 400 sqft per BHK — unusually compact"))
    elif total_sqft / bhk > 800:
        insights.append(("pill-green", "✅ Spacious layout — more than 800 sqft per BHK"))

    if is_ready == 0:
        insights.append(("pill-blue", "🏗️ Under construction — typically 10–15% cheaper than ready-to-move"))
    else:
        insights.append(("pill-green", "✅ Ready to move — immediate possession"))

    if result['price_per_sqft'] > 8000:
        insights.append(("pill-red", "🔴 Premium locality — above Bangalore average rate"))
    elif result['price_per_sqft'] < 4000:
        insights.append(("pill-green", "🟢 Budget-friendly locality"))

    if area_type == "Carpet Area":
        insights.append(("pill-blue", "📐 Carpet area — actual usable space, no common area included"))

    pills_html = "".join([f'<span class="insight-pill {cls}">{msg}</span>' for cls, msg in insights])
    st.markdown(pills_html, unsafe_allow_html=True)

    # Why bathrooms are not a feature
    st.markdown('<p class="section-header">Feature Explanation</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="why-box">
        <strong>Why bathrooms are excluded from this model:</strong><br>
        In the Bangalore dataset, the number of bathrooms is almost perfectly correlated with BHK
        (2BHK → 2 baths, 3BHK → 3 baths in 85%+ of listings). Including it adds
        <em>redundant information</em> that doesn't help the model — and can even hurt accuracy
        by introducing multicollinearity. The real price drivers are <strong>location</strong>
        (accounts for ~55% of variance), <strong>total sqft</strong> (~25%), and
        <strong>BHK configuration</strong> (~12%).
    </div>
    """, unsafe_allow_html=True)

# ── TAB 2: EDA ───────────────────────────────────────────────
with tab2:
    st.markdown("### Exploratory Data Analysis")
    st.markdown("Charts generated during training. Run `python train_model.py` to regenerate them.")

    asset_files = {
        "assets/price_distribution.png": "Price Distribution",
        "assets/top_locations.png": "Top Locations by Price",
        "assets/bhk_vs_price.png": "BHK vs Price",
        "assets/sqft_vs_price.png": "Sqft vs Price",
        "assets/correlation_heatmap.png": "Feature Correlation",
        "assets/area_type_price.png": "Area Type vs Price",
    }

    found = [(path, title) for path, title in asset_files.items() if os.path.exists(path)]
    if not found:
        st.info("No EDA charts found. Run `python train_model.py` to generate them.")
    else:
        for i in range(0, len(found), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(found):
                    path, title = found[i + j]
                    with col:
                        st.markdown(f"**{title}**")
                        st.image(path, use_column_width=True)

# ── TAB 3: MODEL REPORT ──────────────────────────────────────
with tab3:
    st.markdown("### Model Training Report")

    if report:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Model", report["best_model"])
        c2.metric("R² Score", report["best_metrics"]["R² (Test)"])
        c3.metric("MAE", f"₹ {report['best_metrics']['MAE (Lakhs)']} L")

        st.markdown("#### All Models Compared")
        results_df = pd.DataFrame(report["all_results"]).T.reset_index()
        results_df.columns = ["Model"] + list(results_df.columns[1:])
        st.dataframe(results_df, use_container_width=True)

        if report.get("feature_importance_top20"):
            st.markdown("#### Top Feature Importances")
            fi = report["feature_importance_top20"]
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).head(15)
            st.bar_chart(fi_df.set_index("Feature"))

        st.markdown("#### Training Info")
        st.json({
            "Trained at": report["trained_at"],
            "Train samples": report["train_samples"],
            "Test samples": report["test_samples"],
            "Total features": report["num_features"],
            "Unique locations": report["num_locations"],
        })
    else:
        st.info("Model report not found. Run training first.")