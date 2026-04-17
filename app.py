# app.py — Bangalore House Price Predictor (Streamlit)

import streamlit as st
import numpy as np
import pickle

# ─────────────────────────────────────────
# Load model, encoder, locations
# ─────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("location_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("locations.pkl", "rb") as f:
    locations = pickle.load(f)

# ─────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏡",
    layout="centered"
)

st.title("🏡 Bangalore House Price Predictor")
st.markdown("Get an instant price estimate for any property in Bangalore!")
st.markdown("---")

# ─────────────────────────────────────────
# Input Fields
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    location  = st.selectbox("📍 Location", options=locations)
    total_sqft = st.number_input("📐 Total Square Feet", min_value=300, max_value=10000, value=1000, step=50)

with col2:
    bhk  = st.selectbox("🛏️ BHK (Bedrooms)", options=[1, 2, 3, 4, 5, 6])
    bath = st.selectbox("🚿 Bathrooms",       options=[1, 2, 3, 4, 5, 6])

st.markdown("---")

# ─────────────────────────────────────────
# Predict
# ─────────────────────────────────────────
if st.button("🔍 Predict Price"):

    # Encode location
    if location in le.classes_:
        location_encoded = le.transform([location])[0]
    else:
        location_encoded = le.transform(["other"])[0]

    # Prepare input
    input_data = np.array([[location_encoded, total_sqft, bath, bhk]])

    # Predict
    predicted_price = model.predict(input_data)[0]

    st.success(f"### 💰 Estimated Price: ₹ {predicted_price:.2f} Lakhs")
    st.info(f"≈ ₹ {predicted_price * 100000:,.0f}")
    st.balloons()

    # Show price breakdown
    st.markdown("### 📊 Price Breakdown")
    col3, col4, col5 = st.columns(3)
    col3.metric("Total (Lakhs)",   f"₹ {predicted_price:.2f} L")
    col4.metric("Per Sq Ft",       f"₹ {(predicted_price * 100000 / total_sqft):,.0f}")
    col5.metric("BHK",             f"{bhk} BHK")

st.markdown("---")
st.caption("Built with ❤️ using Python, Scikit-learn & Streamlit | Bangalore Housing Dataset")