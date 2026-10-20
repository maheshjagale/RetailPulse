"""
RetailPulse - Feature Engineering Unit Tests
Validates RFM computations, demand features, and inventory metrics.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.rfm import calculate_rfm
from src.features.inventory_features import compute_inventory_metrics
from src.features.demand_features import create_demand_features


@pytest.fixture
def clean_sample_data():
    return pd.DataFrame({
        "Invoice": ["1001", "1002", "1003", "1004"],
        "StockCode": ["A100", "A100", "B200", "A100"],
        "Description": ["WIDGET A", "WIDGET A", "GADGET B", "WIDGET A"],
        "Quantity": [10, 5, 2, 8],
        "Price": [5.0, 5.0, 20.0, 5.0],
        "Revenue": [50.0, 25.0, 40.0, 40.0],
        "Customer ID": [1.0, 1.0, 2.0, 1.0],
        "InvoiceDate": pd.to_datetime(["2021-01-01", "2021-01-10", "2021-01-12", "2021-01-15"]),
        "Date": pd.to_datetime(["2021-01-01", "2021-01-10", "2021-01-12", "2021-01-15"]),
        "IsCancellation": [False, False, False, False]
    })


def test_rfm_calculation(clean_sample_data):
    rfm = calculate_rfm(clean_sample_data, reference_date="2021-01-16")
    cust1 = rfm[rfm["Customer ID"] == 1.0].iloc[0]
    assert cust1["Frequency"] == 3
    assert cust1["Monetary"] == pytest.approx(115.0)
    assert cust1["Recency"] == 1
    assert "Segment" in rfm.columns


def test_inventory_metrics_computation(clean_sample_data):
    inv = compute_inventory_metrics(clean_sample_data, lead_time_days=7, service_level_z=1.645)
    assert len(inv) == 2  # 2 unique SKUs
    item_a = inv[inv["StockCode"] == "A100"].iloc[0]
    assert item_a["LeadTimeDays"] == 7
    assert item_a["SafetyStock"] >= 0
    assert item_a["ReorderPoint"] >= item_a["SafetyStock"]
    assert item_a["EOQ"] >= 1
