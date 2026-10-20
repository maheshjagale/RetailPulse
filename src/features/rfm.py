"""
RetailPulse - RFM Feature Engineering Module
Calculates Recency, Frequency, Monetary values and rule-based segmentation.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def calculate_rfm(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """
    Computes RFM (Recency, Frequency, Monetary) metrics for each customer.
    Filters out cancellations and transactions with non-positive quantities or prices.
    """
    # Filter valid sales with non-null Customer ID
    valid_sales = df[
        (df["Customer ID"].notna()) &
        (df["Quantity"] > 0) &
        (df["Price"] > 0)
    ].copy()

    if not pd.api.types.is_datetime64_any_dtype(valid_sales["InvoiceDate"]):
        valid_sales["InvoiceDate"] = pd.to_datetime(valid_sales["InvoiceDate"])

    if "Revenue" not in valid_sales.columns:
        valid_sales["Revenue"] = valid_sales["Quantity"] * valid_sales["Price"]

    if reference_date is None:
        reference_date = valid_sales["InvoiceDate"].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    logger.info(f"Computing RFM metrics using reference date: {reference_date}")

    # Aggregate by Customer ID
    rfm = valid_sales.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda dates: (reference_date - dates.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        TotalItems=("Quantity", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max")
    ).reset_index()

    # Calculate Customer Tenure in days
    rfm["TenureDays"] = (reference_date - rfm["FirstPurchase"]).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / np.maximum(rfm["Frequency"], 1)

    # Assign R, F, M quantile scores (1 to 5)
    # Recency: lower days is better -> higher score (5)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)

    # Frequency: higher is better (use rank to handle identical values)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Monetary: higher is better
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    rfm["RFM_Index"] = rfm[["R_Score", "F_Score", "M_Score"]].mean(axis=1)

    # 6-8 Meaningful Business Segments
    def assign_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "Recent / New Customers"
        elif r >= 3 and f <= 2 and m >= 3:
            return "Potential Loyalists"
        elif r == 2 and f >= 2:
            return "At Risk"
        elif r <= 2 and f >= 3:
            return "Can't Lose Them"
        elif r == 2 and f <= 2:
            return "About to Sleep"
        else:
            return "Hibernating / Lost"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)
    logger.info(f"RFM calculated for {len(rfm)} unique customers.")
    return rfm
