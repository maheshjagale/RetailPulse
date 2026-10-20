"""
RetailPulse - Inventory Feature Engineering & Optimization Calculations
Computes Safety Stock, Reorder Point (ROP), and Economic Order Quantity (EOQ).
"""
import pandas as pd
import numpy as np
import logging
import config

logger = logging.getLogger(__name__)


def compute_inventory_metrics(
    df: pd.DataFrame,
    lead_time_days: int = config.DEFAULT_LEAD_TIME_DAYS,
    service_level_z: float = config.SERVICE_LEVEL_Z,
    order_cost: float = config.ORDER_COST_FIXED,
    holding_rate: float = config.HOLDING_COST_RATE
) -> pd.DataFrame:
    """
    Computes inventory optimization parameters per product:
    - Average daily demand & demand volatility (std)
    - Lead time demand
    - Safety Stock (SS)
    - Reorder Point (ROP)
    - Economic Order Quantity (EOQ)
    - Stock health status
    """
    valid = df[(df["Quantity"] > 0) & (df["Price"] > 0) & (df["StockCode"].notna())].copy()

    if not pd.api.types.is_datetime64_any_dtype(valid["InvoiceDate"]):
        valid["InvoiceDate"] = pd.to_datetime(valid["InvoiceDate"])

    valid["Date"] = valid["InvoiceDate"].dt.normalize()

    # Get most recent Description and Unit Price per StockCode
    product_meta = valid.sort_values("InvoiceDate").groupby("StockCode").agg(
        Description=("Description", "last"),
        UnitPrice=("Price", "mean"),
        TotalSold=("Quantity", "sum"),
        FirstSale=("InvoiceDate", "min"),
        LastSale=("InvoiceDate", "max")
    ).reset_index()

    # Calculate daily demand per product
    daily_prod = valid.groupby(["StockCode", "Date"])["Quantity"].sum().reset_index()

    # Full active days for each product
    demand_stats = daily_prod.groupby("StockCode")["Quantity"].agg(
        AvgDailyDemand="mean",
        StdDailyDemand="std",
        MaxDailyDemand="max",
        ActiveSalesDays="count"
    ).reset_index().fillna({"StdDailyDemand": 0})

    inv = pd.merge(product_meta, demand_stats, on="StockCode", how="inner")

    # Lead time demand & standard deviation
    inv["LeadTimeDays"] = lead_time_days
    inv["LeadTimeDemand"] = inv["AvgDailyDemand"] * lead_time_days
    inv["LeadTimeStd"] = np.sqrt(lead_time_days) * inv["StdDailyDemand"]

    # Safety Stock = Z * LeadTimeStd
    inv["SafetyStock"] = np.ceil(service_level_z * inv["LeadTimeStd"]).astype(int)

    # Reorder Point = LeadTimeDemand + SafetyStock
    inv["ReorderPoint"] = np.ceil(inv["LeadTimeDemand"] + inv["SafetyStock"]).astype(int)

    # Holding cost per unit per year (H)
    inv["AnnualHoldingCost"] = np.maximum(inv["UnitPrice"] * holding_rate, 0.50)  # min £0.50/yr

    # Annual demand (D)
    inv["AnnualDemand"] = inv["AvgDailyDemand"] * 365

    # EOQ = sqrt(2 * D * S / H)
    inv["EOQ"] = np.ceil(
        np.sqrt((2 * inv["AnnualDemand"] * order_cost) / inv["AnnualHoldingCost"])
    ).fillna(1).astype(int)

    # Current Stock simulation (realistic distribution centered around ROP for demonstration)
    # Using deterministic seed based on StockCode hash for reproducibility
    np.random.seed(42)
    stock_factors = np.random.uniform(0.3, 1.8, size=len(inv))
    inv["CurrentStock"] = np.round(inv["ReorderPoint"] * stock_factors).astype(int)

    # Stock Status Classification
    def classify_status(row):
        curr, rop, ss = row["CurrentStock"], row["ReorderPoint"], row["SafetyStock"]
        if curr <= ss:
            return "Critical / Stockout Risk"
        elif curr <= rop:
            return "Reorder Needed"
        elif curr > (rop + row["EOQ"]):
            return "Overstocked"
        else:
            return "Healthy Stock"

    inv["StockStatus"] = inv.apply(classify_status, axis=1)

    # Suggested Reorder Quantity
    inv["SuggestedReorderQty"] = np.where(
        inv["CurrentStock"] <= inv["ReorderPoint"],
        np.maximum(inv["EOQ"], inv["ReorderPoint"] - inv["CurrentStock"] + inv["SafetyStock"]),
        0
    )

    logger.info(f"Inventory metrics computed for {len(inv)} items.")
    return inv
