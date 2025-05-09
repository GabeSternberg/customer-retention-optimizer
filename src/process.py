"""Data ingestion, cleaning, feature engineering, outlier detection, and clustering."""

import os
import sys
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "customer_features.parquet"


def get_spark() -> SparkSession:
    """Create a local Spark session."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("CustomerRetention")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_and_clean(spark: SparkSession) -> DataFrame:
    """Load the Excel file into Spark and apply cleaning rules."""
    print("Loading Excel file with pandas ...")
    pdf = pd.read_excel(RAW_FILE, engine="openpyxl")
    pdf.columns = [c.strip().replace(" ", "_") for c in pdf.columns]

    # Rename columns for consistency
    rename_map = {
        "Customer_ID": "CustomerID",
        "Invoice": "InvoiceNo",
        "StockCode": "StockCode",
        "Description": "Description",
        "Quantity": "Quantity",
        "InvoiceDate": "InvoiceDate",
        "Price": "UnitPrice",
        "Country": "Country",
    }
    pdf.rename(columns={k: v for k, v in rename_map.items() if k in pdf.columns}, inplace=True)

    sdf = spark.createDataFrame(pdf)

    # Clean
    sdf = (
        sdf
        .filter(F.col("CustomerID").isNotNull())
        .filter(~F.col("InvoiceNo").cast("string").startswith("C"))  # remove cancellations
        .filter(F.col("Quantity") > 0)
        .filter(F.col("UnitPrice") > 0)
        .withColumn("Revenue", (F.col("Quantity") * F.col("UnitPrice")).cast(DoubleType()))
    )
    row_count = sdf.count()
    print(f"Cleaned transactions: {row_count:,} rows")
    return sdf


def build_features(sdf: DataFrame) -> pd.DataFrame:
    """Aggregate per-customer features using Spark."""
    print("Building customer features ...")

    max_date = sdf.agg(F.max("InvoiceDate")).collect()[0][0]
    cutoff_90 = max_date - pd.Timedelta(days=90)
    cutoff_churn = max_date - pd.Timedelta(days=60)

    # Total aggregates
    cust = (
        sdf.groupBy("CustomerID")
        .agg(
            F.datediff(F.lit(max_date), F.max("InvoiceDate")).alias("recency_days"),
            F.countDistinct("InvoiceNo").alias("frequency_total"),
            F.sum("Revenue").alias("monetary_total"),
            (F.sum("Revenue") / F.countDistinct("InvoiceNo")).alias("avg_basket"),
            F.countDistinct("StockCode").alias("unique_products"),
            F.max("InvoiceDate").alias("last_purchase"),
        )
    )

    # 90-day window aggregates
    recent = sdf.filter(F.col("InvoiceDate") >= F.lit(cutoff_90))
    cust_90 = (
        recent.groupBy("CustomerID")
        .agg(
            F.countDistinct("InvoiceNo").alias("frequency_90d"),
            F.sum("Revenue").alias("monetary_90d"),
        )
    )

    # Return rate (invoices with returns / total invoices) — proxy via negative-qty rows in raw
    # Since we already removed cancellations, return_rate is set to 0 as baseline;
    # a more nuanced version would re-read the raw data.
    cust = cust.join(cust_90, on="CustomerID", how="left")
    cust = (
        cust
        .fillna(0, subset=["frequency_90d", "monetary_90d"])
        .withColumn("return_rate", F.lit(0.0))
        .withColumn(
            "churned",
            F.when(F.col("last_purchase") < F.lit(cutoff_churn), 1).otherwise(0),
        )
        .drop("last_purchase")
    )

    pdf = cust.toPandas()
    print(f"Customers: {len(pdf):,}  |  Churned: {pdf['churned'].sum():,} "
          f"({pdf['churned'].mean():.1%})")
    return pdf


def detect_outliers(pdf: pd.DataFrame) -> pd.DataFrame:
    """Flag extreme spenders with IsolationForest and clip monetary values."""
    print("Detecting outliers ...")
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    features = pdf[["monetary_total", "frequency_total", "avg_basket"]].fillna(0)
    pdf["outlier_flag"] = (iso.fit_predict(features) == -1).astype(int)

    q99 = pdf["monetary_90d"].quantile(0.99)
    pdf["monetary_90d_clipped"] = pdf["monetary_90d"].clip(upper=q99)
    print(f"Outliers flagged: {pdf['outlier_flag'].sum():,}")
    return pdf


def cluster_customers(pdf: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """KMeans segmentation robust to heavy-tailed monetary features.

    Key steps to prevent singleton / tiny clusters:
      1. Log-transform monetary features (log1p) to compress heavy tails so that
         a few extreme spenders don't each become their own centroid.
      2. Fit KMeans only on non-outlier rows (outlier_flag == 0) so that extreme
         points cannot distort centroids.  Outliers are then *assigned* to the
         nearest cluster via predict().
      3. Post-check: if any cluster has < 20 members, reduce k and refit to
         guarantee every segment is analytically meaningful.
    """
    print(f"Clustering (initial k={n_clusters}) ...")

    feat_cols = ["recency_days", "frequency_90d", "monetary_90d_clipped",
                 "avg_basket", "unique_products"]
    # Monetary-like columns get a log1p transform to tame heavy tails.
    log_cols = {"monetary_90d_clipped", "avg_basket"}

    X_raw = pdf[feat_cols].fillna(0).copy()
    for col in feat_cols:
        if col in log_cols:
            X_raw[col] = np.log1p(X_raw[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)

    # Split into inliers (for fitting) and outliers (assigned after fit).
    inlier_mask = pdf["outlier_flag"].values == 0
    X_inliers = X_scaled[inlier_mask]

    k = n_clusters
    min_cluster_size = 20

    while k >= 2:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_inliers)

        # Assign ALL rows (inliers + outliers) using the inlier-fit centroids.
        labels = km.predict(X_scaled)

        # Post-check: every cluster must have >= min_cluster_size customers.
        unique, counts = np.unique(labels, return_counts=True)
        if counts.min() >= min_cluster_size:
            break
        print(f"  k={k}: smallest cluster has {counts.min()} customers (< {min_cluster_size}), reducing k")
        k -= 1

    pdf["segment_id"] = labels
    print(f"  Final k={k}")
    for sid in range(k):
        cnt = (pdf["segment_id"] == sid).sum()
        print(f"  Segment {sid}: {cnt:,} customers")
    return pdf


def save_features(pdf: pd.DataFrame) -> Path:
    """Save processed features to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(FEATURES_PATH, index=False)
    print(f"Saved features to {FEATURES_PATH}")
    return FEATURES_PATH


def main() -> None:
    spark = get_spark()
    try:
        sdf = load_and_clean(spark)
        pdf = build_features(sdf)
        pdf = detect_outliers(pdf)
        pdf = cluster_customers(pdf)
        save_features(pdf)
    finally:
        spark.stop()
    print("Processing complete.")


if __name__ == "__main__":
    main()
