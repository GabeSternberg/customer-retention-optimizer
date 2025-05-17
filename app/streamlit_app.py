"""Streamlit app for the Customer Retention Optimizer."""

import subprocess
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.parquet"
OFFER_PATH = PROJECT_ROOT / "data" / "outputs" / "offer_plan.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.set_page_config(page_title="Customer Retention Optimizer", layout="wide")
st.title("Customer Retention Optimizer")
st.markdown("End-to-end ML + optimization pipeline for churn prediction and retention offer allocation.")


def run_step(module: str, label: str, extra_args: list[str] | None = None) -> None:
    """Run a pipeline step as a subprocess."""
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    with st.spinner(f"Running {label} ..."):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
    if result.returncode != 0:
        st.error(f"{label} failed:\n```\n{result.stderr}\n```")
    else:
        st.success(f"{label} complete!")
        if result.stdout.strip():
            st.code(result.stdout, language="text")


# --- Sidebar pipeline controls ---
st.sidebar.header("Pipeline Controls")

if st.sidebar.button("1. Download Data"):
    run_step("src.download_data", "Data Download")

if st.sidebar.button("2. Process & Engineer Features"):
    run_step("src.process", "Processing")

if st.sidebar.button("3. Train Churn Model"):
    run_step("src.train", "Model Training")

budget = st.sidebar.number_input("Marketing Budget ($)", min_value=100, max_value=100_000,
                                  value=5000, step=500)
if st.sidebar.button("4. Optimize Offers"):
    run_step("src.optimize", "Optimization", ["--budget", str(budget)])

if st.sidebar.button("5. Generate Reports"):
    run_step("src.report", "Report Generation")

# --- Main area: results dashboard ---
st.markdown("---")

col1, col2 = st.columns(2)

# Feature stats
if FEATURES_PATH.exists():
    df = pd.read_parquet(FEATURES_PATH)
    with col1:
        st.subheader("Customer Overview")
        st.metric("Total Customers", f"{len(df):,}")
        if "churned" in df.columns:
            churn_rate = df["churned"].mean()
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        if "segment_id" in df.columns:
            st.metric("Segments", df["segment_id"].nunique())
        if "outlier_flag" in df.columns:
            st.metric("Outliers Flagged", f"{df['outlier_flag'].sum():,}")

# Offer stats
if OFFER_PATH.exists():
    offer_df = pd.read_csv(OFFER_PATH)
    with col2:
        st.subheader("Optimization Results")
        total_cost = offer_df["offer_cost"].sum()
        total_saved = offer_df["expected_saved_revenue"].sum()
        st.metric("Total Cost", f"${total_cost:,.0f}")
        st.metric("Expected Saved Revenue", f"${total_saved:,.0f}")
        st.metric("Net Benefit", f"${total_saved - total_cost:,.0f}")

    st.subheader("Offer Distribution")
    st.dataframe(
        offer_df["offer"].value_counts().reset_index().rename(
            columns={"index": "Offer Tier", "offer": "Offer Tier", "count": "Customers"}
        ),
        use_container_width=True,
    )

# Figures
st.markdown("---")
st.subheader("Visualizations")

figure_files = {
    "Customer Segments": "segments.png",
    "Outlier Detection": "outliers.png",
    "ROC Curve": "roc_curve.png",
    "Offer Allocation": "offer_allocation.png",
}

fig_cols = st.columns(2)
for idx, (title, fname) in enumerate(figure_files.items()):
    fpath = FIGURES_DIR / fname
    if fpath.exists():
        with fig_cols[idx % 2]:
            st.markdown(f"**{title}**")
            st.image(str(fpath), use_container_width=True)
