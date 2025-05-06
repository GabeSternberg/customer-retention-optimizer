# Customer Retention Optimizer

End-to-end ML + optimization pipeline that predicts customer churn and allocates retention discounts under a marketing budget to maximize expected saved revenue.

## Dataset

[UCI Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) — real transactional data from a UK online retailer (2009–2011, ~1M rows). The pipeline downloads it automatically.

## Pipeline Overview

| Step | Module | What it does |
|------|--------|-------------|
| A | `src.download_data` | Downloads the Excel dataset to `data/raw/` |
| B | `src.process` | Spark cleaning, feature engineering (RFM + extras), IsolationForest outlier detection, KMeans clustering |
| C | `src.train` | Logistic regression churn model with GridSearchCV, logged to MLflow |
| D | `src.optimize` | OR-Tools CP-SAT allocates retention offers under budget constraints |
| E | `src.report` | Generates matplotlib figures to `reports/figures/` |

## Tech Stack

- **PySpark** — data processing (local mode)
- **scikit-learn** — clustering, outlier detection, classification, hyperparameter tuning
- **MLflow** — experiment tracking and model artifact logging
- **OR-Tools CP-SAT** — constrained optimization for offer allocation
- **Streamlit** — interactive dashboard
- **matplotlib** — visualizations

## Quick Start

```bash
# Clone and install
git clone https://github.com/<your-username>/customer-retention-optimizer.git
cd customer-retention-optimizer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the pipeline step by step
python -m src.download_data
python -m src.process
python -m src.train
python -m src.optimize
python -m src.report

# Or run the interactive dashboard
streamlit run app/streamlit_app.py
```

## Project Structure

```
customer-retention-optimizer/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── download_data.py      # Dataset downloader
│   ├── process.py             # Spark ETL + features + outliers + clustering
│   ├── train.py               # Churn model training + MLflow logging
│   ├── optimize.py            # CP-SAT offer allocation
│   └── report.py              # Matplotlib visualizations
├── app/
│   ├── __init__.py
│   └── streamlit_app.py       # Interactive dashboard
├── notebooks/
│   └── 01_end_to_end.ipynb    # Full pipeline notebook
├── data/
│   ├── raw/                   # Downloaded dataset (gitignored)
│   ├── processed/             # Feature parquet (gitignored)
│   └── outputs/               # Offer plan CSV + ROC data (gitignored)
├── reports/
│   └── figures/               # Generated plots
└── mlruns/                    # MLflow tracking (gitignored)
```

## Pipeline Details

### Feature Engineering

Per-customer aggregates computed with PySpark:

| Feature | Description |
|---------|-------------|
| `recency_days` | Days since last purchase |
| `frequency_90d` | Distinct invoices in last 90 days |
| `monetary_90d` | Total revenue in last 90 days |
| `avg_basket` | Average basket size (revenue / invoices) |
| `return_rate` | Fraction of returned orders |
| `unique_products` | Distinct products purchased |

### Churn Definition

A customer is labeled as churned if they made **no purchases in the last 60 days** of the dataset.

### Optimization Model

**Offer tiers:**

| Tier | Cost | Uplift (retention probability increase) |
|------|------|----|
| None | $0 | 0% |
| 5% coupon | $2 | 5% |
| 10% coupon | $5 | 10% |
| 20% coupon | $10 | 18% |

**Objective:** Maximize total expected saved revenue minus cost.

`expected_saved_revenue = monetary_90d * p_churn * uplift`

**Constraints:**
- Total marketing budget <= configurable (default $5,000)
- 20% coupons limited to <= 10% of customers

### Configuration

```bash
# Custom budget and large-coupon cap
python -m src.optimize --budget 10000 --max-large-pct 0.15
```

## Outputs

- `data/outputs/offer_plan.csv` — per-customer offer assignments
- `reports/figures/segments.png` — PCA scatter + segment counts
- `reports/figures/outliers.png` — outlier detection scatter
- `reports/figures/roc_curve.png` — churn model ROC curve
- `reports/figures/offer_allocation.png` — offer distribution + saved revenue

## Requirements

- Python 3.10+
- Java 8+ (required by PySpark)
- No cloud infrastructure needed — everything runs locally

## License

MIT
