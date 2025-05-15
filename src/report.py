"""Generate visualizations and save to reports/figures/."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.parquet"
OFFER_PATH = PROJECT_ROOT / "data" / "outputs" / "offer_plan.csv"
ROC_PATH = PROJECT_ROOT / "data" / "outputs" / "roc_data.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FEATURE_COLS = [
    "recency_days", "frequency_90d", "monetary_90d_clipped",
    "avg_basket", "unique_products",
]


def plot_segments(df: pd.DataFrame, out: Path) -> None:
    """PCA scatter colored by segment + segment counts bar.

    Uses the same log1p transform on monetary features that clustering uses,
    so the PCA projection faithfully represents the space KMeans operated in.
    """
    # Must mirror the transform in process.cluster_customers
    LOG_COLS = {"monetary_90d_clipped", "avg_basket"}
    X = df[FEATURE_COLS].fillna(0).copy()
    for col in FEATURE_COLS:
        if col in LOG_COLS:
            X[col] = np.log1p(X[col])
    X_scaled = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    segments = df["segment_id"].values
    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=segments,
                               cmap="tab10", alpha=0.4, s=8)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Customer Segments (PCA)")
    plt.colorbar(scatter, ax=axes[0], label="Segment")

    # Counts
    counts = df["segment_id"].value_counts().sort_index()
    axes[1].bar(counts.index.astype(str), counts.values, color="steelblue")
    axes[1].set_xlabel("Segment")
    axes[1].set_ylabel("Customers")
    axes[1].set_title("Segment Sizes")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 20, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_outliers(df: pd.DataFrame, out: Path) -> None:
    """Scatter of monetary vs frequency colored by outlier flag."""
    fig, ax = plt.subplots(figsize=(8, 5))
    normal = df[df["outlier_flag"] == 0]
    outlier = df[df["outlier_flag"] == 1]
    ax.scatter(normal["frequency_90d"], normal["monetary_90d"],
               alpha=0.3, s=8, label="Normal", color="steelblue")
    ax.scatter(outlier["frequency_90d"], outlier["monetary_90d"],
               alpha=0.6, s=15, label="Outlier", color="crimson", marker="x")
    ax.set_xlabel("Frequency (90d)")
    ax.set_ylabel("Monetary (90d)")
    ax.set_title("Outlier Detection (IsolationForest)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_roc(out: Path) -> None:
    """Plot the ROC curve from saved data."""
    roc = pd.read_csv(ROC_PATH)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc["fpr"], roc["tpr"], color="darkorange", lw=2, label="Model")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Prediction")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_offers(df: pd.DataFrame, out: Path) -> None:
    """Offer allocation summary plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Offer distribution
    counts = df["offer"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["gray", "gold", "orange", "red"])
    axes[0].set_xlabel("Offer Tier")
    axes[0].set_ylabel("Customers")
    axes[0].set_title("Offer Allocation")
    for i, (label, v) in enumerate(counts.items()):
        axes[0].text(i, v + 10, str(v), ha="center", fontsize=9)

    # Expected saved revenue by offer
    rev = df.groupby("offer")["expected_saved_revenue"].sum().reindex(counts.index)
    axes[1].bar(rev.index, rev.values, color=["gray", "gold", "orange", "red"])
    axes[1].set_xlabel("Offer Tier")
    axes[1].set_ylabel("Expected Saved Revenue ($)")
    axes[1].set_title("Revenue Saved by Offer Tier")
    for i, (label, v) in enumerate(rev.items()):
        axes[1].text(i, v + 5, f"${v:,.0f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    df = pd.read_parquet(FEATURES_PATH)

    plot_segments(df, FIGURES_DIR / "segments.png")
    plot_outliers(df, FIGURES_DIR / "outliers.png")

    if ROC_PATH.exists():
        plot_roc(FIGURES_DIR / "roc_curve.png")
    else:
        print(f"Skipping ROC curve (no {ROC_PATH}). Run training first.")

    if OFFER_PATH.exists():
        offer_df = pd.read_csv(OFFER_PATH)
        plot_offers(offer_df, FIGURES_DIR / "offer_allocation.png")
    else:
        print(f"Skipping offer plot (no {OFFER_PATH}). Run optimization first.")

    print("Report generation complete.")


if __name__ == "__main__":
    main()
