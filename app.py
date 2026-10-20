"""Streamlit dashboard for RetailPulse analytics."""
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src.models.inventory import InventoryOptimizer


st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = config.PROCESSED_DATA_DIR


@st.cache_data(show_spinner=False)
def load_csv(filename: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def load_excel(filename: str) -> pd.DataFrame | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_excel(path)


def show_missing_data(filename: str) -> None:
    st.warning(
        f"`data/processed/{filename}` is not available. "
        "Add the generated output file to the deployment or configure a data source."
    )


def show_overview() -> None:
    st.header("Executive dashboard")
    retail = load_csv(
        "retail_clean.csv",
        ("Invoice", "Customer ID", "Quantity", "Revenue", "YearMonth", "Country", "StockCode", "Description"),
    )
    inventory = load_csv("inventory_recommendations.csv")
    forecast = load_csv("prophet_forecast_results.csv")

    if retail is None:
        show_missing_data("retail_clean.csv")
    else:
        total_revenue = retail["Revenue"].sum()
        orders = retail["Invoice"].nunique()
        customers = retail["Customer ID"].nunique()
        units = retail["Quantity"].clip(lower=0).sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Net revenue", f"£{total_revenue:,.0f}")
        col2.metric("Transactions", f"{orders:,}")
        col3.metric("Customers", f"{customers:,}")
        col4.metric("Units sold", f"{units:,.0f}")

        left, right = st.columns(2)
        if "YearMonth" in retail.columns:
            monthly = (
                retail.groupby("YearMonth")["Revenue"]
                .sum()
                .rename_axis("YearMonth")
                .reset_index(name="Revenue")
            )
            with left:
                st.subheader("Monthly revenue")
                fig = px.bar(
                    monthly,
                    x="YearMonth",
                    y="Revenue",
                    labels={"YearMonth": "Month", "Revenue": "Revenue (£)"},
                )
                st.plotly_chart(fig, use_container_width=True)

        if "Country" in retail.columns:
            countries = (
                retail.groupby("Country")["Revenue"]
                .sum()
                .nlargest(10)
                .rename_axis("Country")
                .reset_index(name="Revenue")
            )
            with right:
                st.subheader("Top markets by revenue")
                fig = px.bar(
                    countries,
                    x="Revenue",
                    y="Country",
                    orientation="h",
                    labels={"Revenue": "Revenue (£)"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

    if inventory is not None and "StockStatus" in inventory.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Inventory status")
            status = inventory["StockStatus"].value_counts().rename_axis("Status").reset_index(name="SKUs")
            st.plotly_chart(px.pie(status, names="Status", values="SKUs", hole=0.45), use_container_width=True)
        with col2:
            st.subheader("Recommended purchase orders")
            if {"SuggestedReorderQty", "UnitPrice"}.issubset(inventory.columns):
                order_value = (inventory["SuggestedReorderQty"] * inventory["UnitPrice"]).sum()
                st.metric("Estimated reorder value", f"£{order_value:,.0f}")
                st.metric("SKUs to reorder", f"{(inventory['SuggestedReorderQty'] > 0).sum():,}")
            else:
                st.info("Inventory data is available, but reorder value columns are missing.")
    elif inventory is None:
        show_missing_data("inventory_recommendations.csv")

    if forecast is not None:
        st.subheader("Demand forecast snapshot")
        show_forecast_chart(forecast)


def show_forecast_chart(forecast: pd.DataFrame) -> None:
    if "Date" not in forecast.columns:
        st.warning("The forecast output does not include a Date column.")
        return

    forecast = forecast.copy()
    forecast["Date"] = pd.to_datetime(forecast["Date"], errors="coerce")
    chart_columns = [
        column
        for column in ("Demand", "Hybrid_Pred", "Forecasted_Demand", "Lower_Bound", "Upper_Bound")
        if column in forecast.columns and forecast[column].notna().any()
    ]
    if not chart_columns:
        st.warning("The forecast output does not contain any populated demand series.")
        return

    st.plotly_chart(
        px.line(
            forecast,
            x="Date",
            y=chart_columns,
            labels={"value": "Units", "variable": "Series"},
        ),
        use_container_width=True,
    )


def show_demand_forecast() -> None:
    st.header("Demand forecasting")
    forecast = load_csv("prophet_forecast_results.csv")
    if forecast is None:
        show_missing_data("prophet_forecast_results.csv")
        return

    show_forecast_chart(forecast)
    st.subheader("Forecast data")
    st.dataframe(forecast, use_container_width=True, hide_index=True)
    st.download_button(
        "Download forecast CSV",
        forecast.to_csv(index=False).encode("utf-8"),
        file_name="RetailPulse_Demand_Forecast.csv",
        mime="text/csv",
    )


def show_customer_segments() -> None:
    st.header("Customer segmentation")
    segments = load_csv("customer_segments.csv")
    if segments is None:
        segments = load_excel("customer_segments.xlsx")
    if segments is None:
        show_missing_data("customer_segments.csv or customer_segments.xlsx")
        return

    if "Segment_Name" not in segments.columns:
        st.error("The segmentation output is missing the Segment_Name column.")
        return

    st.metric("Customer segments", f"{segments['Segment_Name'].nunique():,}")
    if "Monetary" in segments.columns:
        summary = (
            segments.groupby("Segment_Name", as_index=False)
            .agg(Customers=("Segment_Name", "size"), AverageSpend=("Monetary", "mean"))
            .sort_values("Customers", ascending=False)
        )
        left, right = st.columns(2)
        with left:
            st.subheader("Customers by segment")
            st.plotly_chart(
                px.bar(summary, x="Segment_Name", y="Customers", labels={"Segment_Name": "Segment"}),
                use_container_width=True,
            )
        with right:
            st.subheader("Average customer spend")
            st.plotly_chart(
                px.bar(
                    summary,
                    x="Segment_Name",
                    y="AverageSpend",
                    labels={"Segment_Name": "Segment", "AverageSpend": "Average spend (£)"},
                ),
                use_container_width=True,
            )
        st.dataframe(summary.style.format({"AverageSpend": "£{:,.2f}"}), use_container_width=True, hide_index=True)
    else:
        st.info("Monetary values are not present in the segmentation output.")


def show_churn_scanner() -> None:
    st.header("Churn risk scanner")
    churn = load_csv("customer_churn_scored.csv")
    if churn is None:
        show_missing_data("customer_churn_scored.csv")
        return

    if "Risk_Tier" not in churn.columns:
        st.error("The churn output is missing the Risk_Tier column.")
        return

    counts = churn["Risk_Tier"].value_counts().rename_axis("Risk tier").reset_index(name="Customers")
    high_risk = int(churn["Risk_Tier"].eq("High Risk").sum())
    col1, col2 = st.columns(2)
    col1.metric("Customers scored", f"{len(churn):,}")
    col2.metric("High-risk customers", f"{high_risk:,}")
    st.plotly_chart(
        px.bar(counts, x="Risk tier", y="Customers", color="Risk tier"),
        use_container_width=True,
    )
    if "Churn_Probability" in churn.columns:
        st.subheader("Churn probability distribution")
        st.plotly_chart(
            px.histogram(churn, x="Churn_Probability", nbins=20, labels={"Churn_Probability": "Probability"}),
            use_container_width=True,
        )
    st.caption("Customer identifiers are intentionally omitted from this dashboard view.")


def show_inventory() -> None:
    st.header("Inventory optimization")
    inventory = load_csv("inventory_recommendations.csv")
    if inventory is None:
        inventory = load_excel("inventory_recommendations.xlsx")
    if inventory is None:
        show_missing_data("inventory_recommendations.csv or inventory_recommendations.xlsx")
        return

    if "StockStatus" in inventory.columns:
        status = inventory["StockStatus"].value_counts().rename_axis("Status").reset_index(name="SKUs")
        st.plotly_chart(px.bar(status, x="Status", y="SKUs", color="Status"), use_container_width=True)

    lead_time = st.slider("Supplier lead time (days)", min_value=1, max_value=60, value=7)
    demand_uplift = st.slider("Demand uplift (%)", min_value=0, max_value=100, value=0, step=5)
    optimizer = InventoryOptimizer(lead_time_days=lead_time)
    scenario = optimizer.simulate_what_if(
        inventory,
        lead_time_override=lead_time,
        demand_multiplier=1 + demand_uplift / 100,
    )
    if "Sim_OrderQty" in scenario.columns:
        scenario["EstimatedOrderValue"] = scenario["Sim_OrderQty"] * scenario["UnitPrice"]
        col1, col2 = st.columns(2)
        col1.metric("SKUs requiring an order", f"{(scenario['Sim_OrderQty'] > 0).sum():,}")
        col2.metric("Estimated order value", f"£{scenario['EstimatedOrderValue'].sum():,.0f}")

    display_columns = [
        column
        for column in (
            "StockCode",
            "Description",
            "UnitPrice",
            "StockStatus",
            "CurrentStock",
            "ReorderPoint",
            "SuggestedReorderQty",
            "Sim_OrderQty",
            "EstimatedOrderValue",
        )
        if column in scenario.columns
    ]
    styled = scenario[display_columns].style
    for column in ("UnitPrice", "EstimatedOrderValue"):
        if column in display_columns:
            styled = styled.format("£{:,.2f}", subset=[column])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def show_data_status() -> None:
    st.header("Project data status")
    filenames = (
        "retail_clean.csv",
        "customer_segments.csv",
        "customer_churn_scored.csv",
        "inventory_recommendations.csv",
        "prophet_forecast_results.csv",
        "forecast_model_comparison.xlsx",
    )
    rows = []
    for filename in filenames:
        path = DATA_DIR / filename
        rows.append(
            {
                "File": filename,
                "Status": "Available" if path.exists() else "Missing",
                "Size (MB)": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Data directory: {DATA_DIR}")


st.title("RetailPulse")
st.caption("Retail analytics, customer insights, demand forecasting, and inventory decision support.")

page = st.sidebar.radio(
    "Dashboard",
    (
        "Executive overview",
        "Demand forecasting",
        "Customer segmentation",
        "Churn risk",
        "Inventory optimization",
        "Data status",
    ),
)

if page == "Executive overview":
    show_overview()
elif page == "Demand forecasting":
    show_demand_forecast()
elif page == "Customer segmentation":
    show_customer_segments()
elif page == "Churn risk":
    show_churn_scanner()
elif page == "Inventory optimization":
    show_inventory()
else:
    show_data_status()
