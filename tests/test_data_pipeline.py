"""
RetailPulse - Data Pipeline Unit Tests
Validates data ingestion, cleaning rules, cancellation tagging, and revenue calculations.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.clean_data import clean_retail_data


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "Invoice": ["536365", "536365", "C536366", "536367", "536368"],
        "StockCode": ["85123A", "85123A ", "22633", "84934", "POST"],
        "Description": ["WHITE HEART LIGHT", "WHITE HEART LIGHT", "SET 3 CAKE TINS", "ROSE CANDLE", "POSTAGE"],
        "Quantity": [6, 6, -1, 10, 1],
        "InvoiceDate": [
            "2010-12-01 08:26:00",
            "2010-12-01 08:26:00",
            "2010-12-01 08:28:00",
            "2010-12-01 08:34:00",
            "2010-12-01 08:40:00"
        ],
        "Price": [2.55, 2.55, 4.25, 1.65, 18.00],
        "Customer ID": [17850.0, 17850.0, 17850.0, 13047.0, np.nan],
        "Country": ["United Kingdom", "United Kingdom ", "United Kingdom", "United Kingdom", "United Kingdom"]
    })


def test_deduplication(sample_raw_data):
    clean_df = clean_retail_data(sample_raw_data)
    # The first two rows are exact duplicates; one must be removed
    assert len(clean_df) == 4


def test_cancellation_detection(sample_raw_data):
    clean_df = clean_retail_data(sample_raw_data)
    cancel_row = clean_df[clean_df["Invoice"] == "C536366"].iloc[0]
    assert cancel_row["IsCancellation"] == True
    assert cancel_row["TransactionType"] == "Cancellation / Return"


def test_revenue_derivation(sample_raw_data):
    clean_df = clean_retail_data(sample_raw_data)
    normal_sale = clean_df[clean_df["Invoice"] == "536367"].iloc[0]
    assert normal_sale["Revenue"] == pytest.approx(16.50)


def test_text_normalization(sample_raw_data):
    clean_df = clean_retail_data(sample_raw_data)
    for stock_code in clean_df["StockCode"]:
        assert stock_code == stock_code.strip().upper()
