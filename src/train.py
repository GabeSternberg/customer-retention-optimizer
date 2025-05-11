"""Train a churn-prediction model with proper evaluation and leakage prevention.

Run:  python -m src.train
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

# ── Column roles ──────────────────────────────────────────────────────────────
TARGET = "churned"
ID_COLS = ["CustomerID"]

# Leakage: recency_days > 60 ↔ churned == 1  (definitional identity)
LEAKAGE_COLS = ["recency_days"]

# Data-snooping: fitted on full data before any split / constant
DROP_COLS = [
    "return_rate",   # constant 0 — no signal
    "segment_id",    # KMeans fitted on full dataset
    "outlier_flag",  # IsolationForest fitted on full dataset
]

# Clean numeric features (no leakage, available at prediction time)
NUMERIC_FEATURES = [
    "frequency_90d",
    "monetary_90d",
    "avg_basket",
    "unique_products",
    "frequency_total",
    "monetary_total",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def leakage_audit(df: pd.DataFrame) -> None:
    """Print ranked features by target correlation & mutual information."""
    print("\n" + "=" * 70)
    print("LEAKAGE AUDIT")
    print("=" * 70)

    y = df[TARGET]
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors="ignore")

    # Pearson correlation
    corr = numeric.corrwith(y).abs().sort_values(ascending=False)
    print("\nRanked by |Pearson r| with target:")
    for feat, val in corr.items():
        tag = " *** LEAKAGE" if val > 0.95 else (" ** suspicious" if val > 0.80 else "")
        print(f"  {feat:30s}  r = {val:.4f}{tag}")

    # Mutual information
    X_mi = numeric.fillna(0).values
    mi = mutual_info_classif(X_mi, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi, index=numeric.columns).sort_values(ascending=False)
    print("\nRanked by mutual information with target:")
    for feat, val in mi_series.items():
        print(f"  {feat:30s}  MI = {val:.4f}")

    print(f"\n→ Dropping LEAKAGE columns:    {LEAKAGE_COLS}")
    print(f"→ Dropping DATA-SNOOP columns: {DROP_COLS}")
    print(f"→ Dropping ID columns:         {ID_COLS}")
    print(f"→ Using clean features:        {NUMERIC_FEATURES}")
    print("=" * 70 + "\n")


def build_pipeline() -> Pipeline:
    """sklearn Pipeline + ColumnTransformer; all fitting happens on train only."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])


def evaluate(model, X, y, label="Model"):
    """Score a fitted model and return metrics dict."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    print(f"  [{label}]  AUC={auc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return {"roc_auc": auc, "precision": prec, "recall": rec, "f1": f1,
            "y_prob": y_prob, "y_pred": y_pred}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & audit ─────────────────────────────────────────────────────
    print("Loading features ...")
    df = load_data()
    leakage_audit(df)

    X = df[NUMERIC_FEATURES].copy()
    y = df[TARGET].values
    print(f"Samples: {len(y):,}  |  Churn rate: {y.mean():.2%}\n")

    # 2. Stratified train / test split FIRST ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"X_train: {X_train.shape}   churn rate: {y_train.mean():.2%}")
    print(f"X_test:  {X_test.shape}   churn rate: {y_test.mean():.2%}")

    # 3. GridSearchCV on training set only ────────────────────────────────
    print("\nGrid search (5-fold stratified CV on training set) ...")
    pipe = build_pipeline()
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    cv_mean = gs.best_score_
    cv_std = gs.cv_results_["std_test_score"][gs.best_index_]
    print(f"Best params: {gs.best_params_}")
    print(f"CV AUC (mean +/- std): {cv_mean:.4f} +/- {cv_std:.4f}")

    # 4. Held-out test evaluation ─────────────────────────────────────────
    print("\n── Test-set evaluation (untouched holdout) ──")
    test_res = evaluate(best, X_test, y_test, label="Test")

    # 5. Sanity checks ────────────────────────────────────────────────────
    print("\n── Sanity checks ──")

    # 5a. Dummy baseline (stratified random)
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    y_prob_dummy = dummy.predict_proba(X_test)[:, 1]
    auc_dummy = roc_auc_score(y_test, y_prob_dummy)
    print(f"  [Dummy stratified]  AUC = {auc_dummy:.4f}")

    # 5b. Shuffled-label check (should ≈ 0.5)
    rng = np.random.RandomState(RANDOM_STATE)
    y_shuffled = rng.permutation(y_train)
    pipe_shuf = build_pipeline()
    pipe_shuf.set_params(**gs.best_params_)
    pipe_shuf.fit(X_train, y_shuffled)
    y_prob_shuf = pipe_shuf.predict_proba(X_test)[:, 1]
    auc_shuf = roc_auc_score(y_test, y_prob_shuf)
    print(f"  [Shuffled labels]   AUC = {auc_shuf:.4f}")

    # 6. Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve (test set)
    fpr, tpr, _ = roc_curve(y_test, test_res["y_prob"])
    axes[0].plot(fpr, tpr, lw=2, color="darkorange",
                 label=f"Model (AUC = {test_res['roc_auc']:.3f})")
    fpr_d, tpr_d, _ = roc_curve(y_test, y_prob_dummy)
    axes[0].plot(fpr_d, tpr_d, lw=1, color="green", linestyle=":",
                 label=f"Dummy (AUC = {auc_dummy:.3f})")
    axes[0].plot([0, 1], [0, 1], lw=1, color="gray", linestyle="--", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Churn Prediction (Test Set)")
    axes[0].legend(loc="lower right")

    # Confusion matrix (default threshold = 0.5)
    cm = confusion_matrix(y_test, test_res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Active", "Churned"])
    disp.plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("Confusion Matrix (threshold = 0.5)")

    plt.tight_layout()
    roc_path = OUTPUT_DIR / "roc.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved ROC + confusion matrix → {roc_path}")

    # Persist roc_data.csv for report.py compatibility
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(OUTPUT_DIR / "roc_data.csv", index=False)

    # 7. Save predictions for downstream (optimize.py) ────────────────────
    # Use the final pipeline (trained on train set) to score ALL customers
    df["p_churn"] = best.predict_proba(X[NUMERIC_FEATURES])[:, 1]
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"Saved predictions → {FEATURES_PATH}")

    # 8. MLflow ───────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("churn-prediction")
    with mlflow.start_run(run_name="logistic_regression_fixed") as run:
        mlflow.log_params({
            "model": "LogisticRegression",
            "features": ",".join(NUMERIC_FEATURES),
            "dropped_leakage": ",".join(LEAKAGE_COLS),
            "dropped_snooping": ",".join(DROP_COLS),
            **{k: str(v) for k, v in gs.best_params_.items()},
        })
        mlflow.log_metrics({
            "cv_auc_mean": cv_mean,
            "cv_auc_std": cv_std,
            "test_roc_auc": test_res["roc_auc"],
            "test_precision": test_res["precision"],
            "test_recall": test_res["recall"],
            "test_f1": test_res["f1"],
            "dummy_auc": auc_dummy,
            "shuffled_auc": auc_shuf,
        })
        mlflow.sklearn.log_model(best, "churn_model")
        print(f"MLflow run: {run.info.run_id}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  CV AUC (5-fold, train only):   {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"  Test AUC (held-out {TEST_SIZE:.0%}):      {test_res['roc_auc']:.4f}")
    print(f"  Dummy baseline AUC:            {auc_dummy:.4f}")
    print(f"  Shuffled-labels AUC:           {auc_shuf:.4f}")
    print()
    print("LEAKAGE FOUND:")
    print("  1. recency_days has r ≈ 1.0 with target because churned is DEFINED")
    print("     as recency_days > 60.  This feature alone gives AUC ≈ 1.0.")
    print("  2. segment_id / outlier_flag were computed by fitting KMeans /")
    print("     IsolationForest on the FULL dataset before any split (data snooping).")
    print("  3. return_rate was constant 0.0 — zero signal.")
    print("  4. Original code evaluated predict_proba(X) on the SAME X used for")
    print("     fit(X, y) — no holdout set — inflating all reported metrics.")
    print()
    print("FIXES APPLIED:")
    print("  - Dropped recency_days, segment_id, outlier_flag, return_rate")
    print("  - train_test_split(stratify=y) BEFORE any fitting")
    print("  - sklearn Pipeline + ColumnTransformer (impute + scale on train only)")
    print("  - GridSearchCV on training set; final metrics on held-out test set")
    print("  - Sanity checks: dummy baseline + shuffled-label AUC ≈ 0.5")
    print("=" * 70)
    print("Training complete.")


if __name__ == "__main__":
    main()
