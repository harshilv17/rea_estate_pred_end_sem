import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model (sklearn Pipeline from the notebook)
model = pickle.load(open("models/price_model.pkl", "rb"))

# Load dataset to get locations & area types
data = pd.read_csv("data/Bengaluru_House_Data.csv")

# --- Pre-process data the same way as the notebook to get valid dropdown values ---
data = data.drop(['society', 'availability'], axis=1, errors='ignore')
data = data.dropna()

# Extract BHK
extracted_bhk_str = data['size'].str.split().str[0]
data['bhk'] = pd.to_numeric(extracted_bhk_str, errors='coerce')
data.dropna(subset=['bhk'], inplace=True)
data['bhk'] = data['bhk'].astype(int)
data = data.drop('size', axis=1)

# Convert total_sqft
def convert_sqft(x):
    try:
        if '-' in str(x):
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft)
data = data.dropna()

st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(17, 153, 142, 0.3);
        margin: 1rem 0;
    }

    .prediction-card h2 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .prediction-card p {
        font-size: 1rem;
        opacity: 0.9;
    }

    .info-card {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }

    .stat-label {
        color: #888;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stat-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    div[data-testid="stSidebar"] label {
        color: #e0e0ff !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏠 Intelligent Property Price Prediction</h1>
    <p>AI-powered price estimation for Bengaluru real estate</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.markdown("## 🔧 Property Features")
st.sidebar.markdown("---")

area_type = st.sidebar.selectbox(
    "🏗️ Area Type",
    sorted(data["area_type"].unique())
)

location = st.sidebar.selectbox(
    "📍 Location",
    sorted(data["location"].dropna().unique())
)

total_sqft = st.sidebar.number_input(
    "📐 Total Square Feet",
    min_value=300,
    max_value=10000,
    value=1200
)

bath = st.sidebar.number_input(
    "🚿 Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

balcony = st.sidebar.number_input(
    "🌿 Number of Balconies",
    min_value=0,
    max_value=5,
    value=1
)

bhk = st.sidebar.number_input(
    "🛏️ Number of Bedrooms (BHK)",
    min_value=1,
    max_value=10,
    value=2
)

# Main content area — dataset stats
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="stat-label">Total Properties in Dataset</div>
        <div class="stat-value">{:,}</div>
    </div>
    """.format(len(data)), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="stat-label">Unique Locations</div>
        <div class="stat-value">{:,}</div>
    </div>
    """.format(data["location"].nunique()), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="stat-label">Average Price</div>
        <div class="stat-value">₹ {:.1f} L</div>
    </div>
    """.format(data["price"].mean()), unsafe_allow_html=True)

st.markdown("---")

# Summary of selected features
st.subheader("📋 Selected Property Details")
detail_cols = st.columns(3)
with detail_cols[0]:
    st.metric("Area Type", area_type)
    st.metric("Bathrooms", bath)
with detail_cols[1]:
    loc_display = location[:25] + "..." if len(str(location)) > 25 else location
    st.metric("Location", loc_display)
    st.metric("Balconies", balcony)
with detail_cols[2]:
    st.metric("Total Sqft", f"{total_sqft:,}")
    st.metric("BHK", bhk)

st.markdown("---")

# Predict Button
if st.button("🔮 Predict Property Price", use_container_width=True, type="primary"):

    # Build input DataFrame with the SAME columns the pipeline expects
    input_data = pd.DataFrame({
        "area_type": [area_type],
        "location": [location],
        "total_sqft": [float(total_sqft)],
        "bath": [float(bath)],
        "balcony": [float(balcony)],
        "bhk": [int(bhk)]
    })

    prediction = model.predict(input_data)[0]

    st.markdown("""
    <div class="prediction-card">
        <p>Estimated Property Price</p>
        <h2>₹ {:.2f} Lakhs</h2>
        <p>Based on historical Bengaluru housing data</p>
    </div>
    """.format(prediction), unsafe_allow_html=True)

    # Price per sqft
    price_per_sqft = (prediction * 100000) / total_sqft
    st.info(f"💡 **Price per sq.ft:** ₹ {price_per_sqft:,.0f}")

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #888;">
        <p style="font-size: 3rem;">👆</p>
        <p style="font-size: 1.2rem;">Configure property details in the sidebar and click <b>Predict</b></p>
    </div>
    """, unsafe_allow_html=True)