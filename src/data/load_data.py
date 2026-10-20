"""
RetailPulse - Data Loading Module
Handles ingestion of raw and processed retail transaction data.
"""
import os
import logging
from pathlib import Path
import pandas as pd
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_raw_data(file_path: Path = config.RAW_DATA_FILE, sheet_name=None) -> pd.DataFrame:
    """
    Ingests the raw Online Retail II dataset from Excel.
    If sheet_name is None, loads the default sheet or combines sheets if multiple exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    logger.info(f"Loading raw data from {file_path}...")
    if sheet_name:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        # Load first sheet or specific sheet
        excel_file = pd.ExcelFile(file_path)
        logger.info(f"Available sheets: {excel_file.sheet_names}")
        df = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])

    logger.info(f"Raw data loaded successfully. Shape: {df.shape}")
    return df


def load_clean_data(file_path: Path = config.CLEAN_DATA_CSV) -> pd.DataFrame:
    """
    Loads pre-processed clean retail data from CSV for high performance.
    Parses datetime columns automatically.
    """
    if not os.path.exists(file_path):
        # Fallback to excel if csv not found
        if os.path.exists(config.CLEAN_DATA_XLSX):
            logger.info(f"Loading clean data from Excel: {config.CLEAN_DATA_XLSX}")
            df = pd.read_excel(config.CLEAN_DATA_XLSX)
        else:
            raise FileNotFoundError(f"Clean data file not found at: {file_path}")
    else:
        logger.info(f"Loading clean data from CSV: {file_path}")
        df = pd.read_csv(file_path)

    # Convert date columns
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    logger.info(f"Clean data loaded successfully. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = load_clean_data()
    print("Columns:", df.columns.tolist())
    print("Row count:", len(df))
