
import streamlit as st
import pandas as pd
from pathlib import Path

# Project configuration
st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Dashboard title
st.title("RetailPulse")
st.subheader("Retail Analytics & Decision Support")

st.write(
    "Explore customer segments, demand forecasts, "
    "and inventory recommendations."
)

# Load existing project outputs
@st.cache_data
def load_excel(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_excel(path)

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)

customer_segments = load_excel("customer_segments.xlsx")
inventory = load_excel("inventory_recommendations.xlsx")
forecast = load_csv("prophet_forecast_results.csv")
forecast_comparison = load_excel("forecast_model_comparison.xlsx")

# Display file status
st.header("Project Data Status")

files = {
    "Customer Segments": customer_segments,
    "Inventory Recommendations": inventory,
    "Prophet Forecast": forecast,
    "Forecast Model Comparison": forecast_comparison,
}

for name, data in files.items():
    if data is not None:
        st.success(f"{name}: loaded successfully ({len(data)} rows)")
    else:
        st.warning(f"{name}: file not found in data/processed/")
