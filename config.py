"""
RetailPulse - Configuration Module
Centralized settings, file paths, business parameters, and model hyperparameter defaults.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DASHBOARD_DIR = BASE_DIR / "dashboard"
TESTS_DIR = BASE_DIR / "tests"

# Ensure runtime directories exist
for directory in [PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR, DASHBOARD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "online_retail_II.xlsx"
CLEAN_DATA_CSV = PROCESSED_DATA_DIR / "retail_clean.csv"
CLEAN_DATA_XLSX = PROCESSED_DATA_DIR / "retail_clean.xlsx"
CUSTOMER_SEGMENTS_CSV = PROCESSED_DATA_DIR / "customer_segments.csv"
CUSTOMER_SEGMENTS_XLSX = PROCESSED_DATA_DIR / "customer_segments.xlsx"
FORECAST_RESULTS_CSV = PROCESSED_DATA_DIR / "prophet_forecast_results.csv"
INVENTORY_RECOMMENDATIONS_XLSX = PROCESSED_DATA_DIR / "inventory_recommendations.xlsx"

# Serialized Model Artifacts
CHURN_MODEL_FILE = MODELS_DIR / "churn_model.pkl"
FORECAST_MODEL_FILE = MODELS_DIR / "forecast_model.pkl"
SEGMENTATION_MODEL_FILE = MODELS_DIR / "segmentation_model.pkl"

# Business & Optimization Constants
DEFAULT_LEAD_TIME_DAYS = 7
SERVICE_LEVEL_Z = 1.645  # 95% service level
ORDER_COST_FIXED = 50.0  # £ per purchase order (S)
HOLDING_COST_RATE = 0.20  # 20% annual holding cost rate (H = rate * unit price)
CHURN_OBSERVATION_DAYS = 90  # Churn inactivity threshold window
FORECAST_HORIZON_DAYS = 30  # Forecast 30-day ahead

# Candidate Metadata for Documentation & Report
CANDIDATE_NAME = "Mahesh Jagale"
PROJECT_TITLE = "RetailPulse – Customer Analytics & Demand Forecasting Platform"
SUBMISSION_WINDOW = "20th Sept to 20th Oct 2026"
EDITION = "Zidio Development - Data Science & Analytics Domain (v2.0 Industry Edition)"
