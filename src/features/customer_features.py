"""
RetailPulse - Advanced Customer Feature Engineering Module
Builds rich behavioral, cadence, and monetary features for Churn Prediction and CLV analysis.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_customer_churn_features(df: pd.DataFrame, cutoff_date: pd.Timestamp, observation_days: int = 365, churn_window_days: int = 90) -> pd.DataFrame:
    """
    Constructs comprehensive behavioral feature matrix for churn prediction.
    Observation window: [cutoff_date - observation_days, cutoff_date)
    Churn window: [cutoff_date, cutoff_date + churn_window_days)
    A customer is considered Churned (1) if they made 0 purchases in the churn window, else (0).
    """
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    if "Revenue" not in df.columns:
        df["Revenue"] = df["Quantity"] * df["Price"]

    obs_start = cutoff_date - pd.Timedelta(days=observation_days)
    churn_end = cutoff_date + pd.Timedelta(days=churn_window_days)

    logger.info(f"Extracting churn features. Obs: [{obs_start.date()} to {cutoff_date.date()}], Churn window: [{cutoff_date.date()} to {churn_end.date()}]")

    # Transactions in observation window
    obs_df = df[(df["InvoiceDate"] >= obs_start) & (df["InvoiceDate"] < cutoff_date) & (df["Customer ID"].notna())].copy()

    # Valid purchases (non-cancellation)
    valid_obs = obs_df[(obs_df["Quantity"] > 0) & (obs_df["Price"] > 0)].copy()

    # Cancellations
    cancellations = obs_df[(obs_df["Quantity"] < 0) | (obs_df["Invoice"].astype(str).str.startswith("C", na=False))].copy()

    # 1. Base RFM & Tenure
    cust_base = valid_obs.groupby("Customer ID").agg(
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique")
    )

    cust_base["Recency"] = (cutoff_date - cust_base["LastPurchase"]).dt.days
    cust_base["TenureDays"] = (cutoff_date - cust_base["FirstPurchase"]).dt.days
    cust_base["AvgOrderValue"] = cust_base["Monetary"] / np.maximum(cust_base["Frequency"], 1)
    cust_base["AvgOrderQuantity"] = cust_base["TotalQuantity"] / np.maximum(cust_base["Frequency"], 1)

    # 2. Inter-purchase time cadence
    order_dates = valid_obs.groupby(["Customer ID", "Invoice"])["InvoiceDate"].min().reset_index()
    order_dates = order_dates.sort_values(["Customer ID", "InvoiceDate"])
    order_dates["PrevOrderDate"] = order_dates.groupby("Customer ID")["InvoiceDate"].shift(1)
    order_dates["DaysBetweenOrders"] = (order_dates["InvoiceDate"] - order_dates["PrevOrderDate"]).dt.days

    cadence = order_dates.groupby("Customer ID").agg(
        MeanDaysBetweenOrders=("DaysBetweenOrders", "mean"),
        StdDaysBetweenOrders=("DaysBetweenOrders", "std"),
        MaxDaysBetweenOrders=("DaysBetweenOrders", "max")
    ).fillna(0)

    # 3. Velocity / Momentum features (Last 60 days vs prior)
    last_60_start = cutoff_date - pd.Timedelta(days=60)
    recent_tx = valid_obs[valid_obs["InvoiceDate"] >= last_60_start]

    recent_metrics = recent_tx.groupby("Customer ID").agg(
        RecentInvoices=("Invoice", "nunique"),
        RecentRevenue=("Revenue", "sum")
    )

    # 4. Cancellation features
    cancel_metrics = cancellations.groupby("Customer ID").agg(
        CancelCount=("Invoice", "nunique"),
        CancelQuantity=("Quantity", lambda x: abs(x.sum()))
    )

    # Merge all feature components
    features = cust_base.join(cadence, how="left")
    features = features.join(recent_metrics, how="left").fillna({"RecentInvoices": 0, "RecentRevenue": 0})
    features = features.join(cancel_metrics, how="left").fillna({"CancelCount": 0, "CancelQuantity": 0})

    # Ratios and momentum
    features["CancelRate"] = features["CancelCount"] / np.maximum(features["Frequency"] + features["CancelCount"], 1)
    features["RecentInvoiceRatio"] = features["RecentInvoices"] / np.maximum(features["Frequency"], 1)
    features["RecentRevenueRatio"] = features["RecentRevenue"] / np.maximum(features["Monetary"], 1)
    features["DaysSinceExpectedOrder"] = np.maximum(0, features["Recency"] - features["MeanDaysBetweenOrders"])

    # 5. Determine ground truth Churn target
    future_df = df[(df["InvoiceDate"] >= cutoff_date) & (df["InvoiceDate"] < churn_end) & (df["Customer ID"].notna())]
    future_valid = future_df[(future_df["Quantity"] > 0) & (future_df["Price"] > 0)]
    active_future_customers = set(future_valid["Customer ID"].unique())

    features["Churn"] = (~features.index.isin(active_future_customers)).astype(int)

    logger.info(f"Feature matrix built. Total samples: {len(features)}. Churn rate: {features['Churn'].mean():.2%}")
    return features.reset_index()
