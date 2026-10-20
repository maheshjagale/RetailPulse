<<<<<<< HEAD
# RetailPulse
=======
# 📊 RetailPulse – Customer Analytics & Demand Forecasting Platform
>>>>>>> c90ff00c0a171c93fe5610f62df11523fdb7e5af

RetailPulse is a retail analytics and forecasting project built to turn transaction history into actionable operational intelligence. It combines customer analytics, demand forecasting, churn prediction, and inventory optimization in a single Streamlit app.

This repository currently reflects the cleaned-up, production-oriented project structure after removing older legacy files and scripts that are no longer used in the active workflow.

---

## Project overview

RetailPulse helps answer questions such as:

- Which customers are likely to churn?
- Which products and categories are driving revenue?
- What demand should we expect over the next 30 days?
- Which SKUs need reordering or have stockout risk?
- How do customer behavior, recency, cadence, and spend vary across segments?

The project is designed around a retail data pipeline and a dashboard that surfaces the generated insights in a business-friendly format.

---

## What is included in this repository

### Core application
- `app.py` – Streamlit dashboard with executive KPIs, demand forecast views, customer segmentation, churn analysis, and inventory recommendations.
- `config.py` – central configuration and business constants.

### Data assets
- `data/raw/online_retail_II.xlsx` – original retail transactional dataset.
- `data/processed/retail_clean.csv` – cleaned retail dataset used by the app.
- `data/processed/customer_segments.csv` – customer cohorts generated from RFM logic.
- `data/processed/customer_churn_scored.csv` – customer churn risk scoring output.
- `data/processed/inventory_recommendations.csv` – inventory health and reorder recommendations.
- `data/processed/prophet_forecast_results.csv` – forecast outputs for demand modeling.
- `data/processed/forecast_model_comparison.xlsx` – model comparison exports.

### Feature engineering and modeling
- `src/data/load_data.py` – data loading utilities.
- `src/data/clean_data.py` – cleaning logic used during data preparation.
- `src/features/rfm.py` – RFM calculations and segment features.
- `src/features/customer_features.py` – customer behavior and cadence features.
- `src/features/demand_features.py` – demand lags and time-based features.
- `src/features/inventory_features.py` – safety stock, reorder point, EOQ, and inventory health logic.
- `src/models/churn.py` – churn predictor model and scoring logic.
- `src/models/forecasting.py` – forecasting model implementation.
- `src/models/segmentation.py` – segmentation modeling logic.
- `src/models/inventory.py` – inventory optimization calculations.

### Notebooks
- `notebooks/01_data_understanding.ipynb`
- `notebooks/02_data_cleaning.ipynb`
- `notebooks/03_eda.ipynb`
- `notebooks/04_customer_segmentation.ipynb`
- `notebooks/05_demand_forecasting.ipynb`
- `notebooks/06_churn_prediction.ipynb`
- `notebooks/07_inventory_optimization.ipynb`

### Supporting files
- `requirements.txt` – Python dependencies for the project.
- `Dockerfile` – containerization setup.
- `docker-compose.yml` – local container orchestration.
- `tests/` – automated validation for data loading, features, and model logic.
- `models/` – serialized model files used in the app and evaluation workflow.

---

## Current capabilities

The dashboard currently supports:

- Executive KPI dashboard
- Monthly revenue and sales trend analysis
- Product-level revenue ranking
- Demand forecasting and scenario analysis
- Customer segmentation using RFM and cohort behavior
- Churn risk analysis with risk-tier output
- Inventory health and reorder recommendations

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/maheshjagale/RetailPulse.git
cd RetailPulse
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
.\venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

---

## Run tests

```bash
python -m pytest tests -q
```

---

## Notes on public data sharing

If this project is published to a public GitHub repository, avoid committing raw customer-level transaction exports or other identifiable customer records unless you explicitly want them to be public.

For public repositories, it is usually safer to keep:

- aggregate forecast output files
- inventory recommendation files
- anonymized or scored customer summaries only when required

Do not include raw transaction files or customer identifiers in a public-facing data repo unless there is a clear privacy and compliance review.

---

## Repository structure

```text
RetailPulse/
├── app.py
├── config.py
├── Dockerfile
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .gitignore
├── .github/
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx
│   └── processed/
│       ├── retail_clean.csv
│       ├── customer_segments.csv
│       ├── customer_churn_scored.csv
│       ├── inventory_recommendations.csv
│       ├── prophet_forecast_results.csv
│       ├── forecast_model_comparison.xlsx
│       └── inventory_recommendations.xlsx
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── notebooks/
├── models/
├── reports/
├── tests/
└── logs/
```

---

## License

This project is intended for data science and analytics use and is structured for local experimentation and portfolio/demo deployment.

---

## Author

Mahesh Jagale
