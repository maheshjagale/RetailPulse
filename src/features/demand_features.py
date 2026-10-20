"""
RetailPulse - Demand Feature Engineering Module
Builds time-series aggregations, calendar features, lags, and rolling statistics for forecasting.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def aggregate_daily_demand(df: pd.DataFrame, stock_code: str = None) -> pd.DataFrame:
    """
    Aggregates transactions into daily demand time series.
    Optionally filters for a specific stock_code (or aggregates all products for macro demand).
    """
    valid = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()

    if not pd.api.types.is_datetime64_any_dtype(valid["InvoiceDate"]):
        valid["InvoiceDate"] = pd.to_datetime(valid["InvoiceDate"])

    if stock_code:
        valid = valid[valid["StockCode"].astype(str) == str(stock_code)]

    valid["Date"] = valid["InvoiceDate"].dt.normalize()

    daily = valid.groupby("Date").agg(
        Demand=("Quantity", "sum"),
        Revenue=("Revenue", "sum") if "Revenue" in valid.columns else ("Quantity", lambda x: 0),
        OrderCount=("Invoice", "nunique"),
        CustomerCount=("Customer ID", "nunique")
    ).reset_index()

    # Reindex to ensure continuous daily date range (fill missing days with 0 demand)
    full_date_range = pd.date_range(start=daily["Date"].min(), end=daily["Date"].max(), freq="D")
    daily = daily.set_index("Date").reindex(full_date_range).fillna(0).reset_index()
    daily.rename(columns={"index": "Date"}, inplace=True)

    return daily


def create_demand_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs calendar, lag, and rolling statistics on daily demand for machine learning forecasting.
    """
    df = daily_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    # Calendar features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfMonth"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Cyclic calendar encoding
    df["Sin_DayOfWeek"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["Cos_DayOfWeek"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["Sin_Month"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Cos_Month"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Lags
    for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
        df[f"Lag_{lag}"] = df["Demand"].shift(lag)

    # Rolling statistics (using shifted series to prevent data leakage)
    for window in [7, 14, 30]:
        df[f"Rolling_Mean_{window}"] = df["Demand"].shift(1).rolling(window=window).mean()
        df[f"Rolling_Std_{window}"] = df["Demand"].shift(1).rolling(window=window).std().fillna(0)
        df[f"Rolling_Max_{window}"] = df["Demand"].shift(1).rolling(window=window).max()
        df[f"Rolling_Min_{window}"] = df["Demand"].shift(1).rolling(window=window).min()

    return df
