"""
RetailPulse - Data Cleaning & Preprocessing Pipeline
Performs deduplication, text normalization, cancellation tagging, revenue derivation, and temporal parsing.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import config
from src.data.load_data import load_raw_data

logger = logging.getLogger(__name__)


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw retail transaction data:
    1. Removes exact duplicates
    2. Normalizes text fields (StockCode, Description, Country)
    3. Flags cancellations (Invoice starting with 'C' or negative Quantity)
    4. Computes Revenue = Quantity * Price
    5. Extracts datetime components (Year, Month, Day, DayOfWeek, YearMonth, Date)
    6. Tags TransactionType (Normal Sale, Free/Sample, Price Adjustment, Return)
    """
    logger.info(f"Starting data cleaning pipeline. Initial shape: {df.shape}")
    df_clean = df.copy()

    # 1. Text normalization
    df_clean["StockCode"] = df_clean["StockCode"].astype(str).str.strip().str.upper()
    if "Description" in df_clean.columns:
        df_clean["Description"] = df_clean["Description"].astype(str).str.strip().str.upper()
    if "Country" in df_clean.columns:
        df_clean["Country"] = df_clean["Country"].astype(str).str.strip()

    # 2. Deduplication
    init_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Removed {init_count - len(df_clean)} duplicate rows.")

    # 3. Cancellation detection
    df_clean["IsCancellation"] = (
        df_clean["Invoice"].astype(str).str.startswith("C") | (df_clean["Quantity"] < 0)
    )

    # 4. Revenue computation
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["Price"]

    # 5. Transaction classification
    df_clean["TransactionType"] = "Normal Sale"
    df_clean.loc[df_clean["IsCancellation"], "TransactionType"] = "Cancellation / Return"
    df_clean.loc[(df_clean["Price"] == 0) & (~df_clean["IsCancellation"]), "TransactionType"] = "Free / Sample / Adjustment"
    df_clean.loc[(df_clean["Price"] < 0), "TransactionType"] = "Debt / Bad Adjustment"

    # 6. Temporal parsing
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["Year"] = df_clean["InvoiceDate"].dt.year
    df_clean["Month"] = df_clean["InvoiceDate"].dt.month
    df_clean["Day"] = df_clean["InvoiceDate"].dt.day
    df_clean["DayOfWeek"] = df_clean["InvoiceDate"].dt.day_name()
    df_clean["YearMonth"] = df_clean["InvoiceDate"].dt.to_period("M").astype(str)
    df_clean["Date"] = df_clean["InvoiceDate"].dt.normalize()

    logger.info(f"Data cleaning complete. Clean shape: {df_clean.shape}")
    return df_clean


def run_etl_pipeline(save_csv: bool = True) -> pd.DataFrame:
    """
    Full ETL runner: loads raw data, executes cleaning, and saves to data/processed/.
    """
    # If clean csv already exists and is valid, load it
    if config.CLEAN_DATA_CSV.exists():
        logger.info(f"Clean data already exists at {config.CLEAN_DATA_CSV}.")
        return pd.read_csv(config.CLEAN_DATA_CSV)

    raw_df = load_raw_data()
    clean_df = clean_retail_data(raw_df)

    if save_csv:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving cleaned dataset to {config.CLEAN_DATA_CSV}...")
        clean_df.to_csv(config.CLEAN_DATA_CSV, index=False)

    return clean_df


if __name__ == "__main__":
    df = run_etl_pipeline()
    print("ETL pipeline executed successfully. Rows:", len(df))
