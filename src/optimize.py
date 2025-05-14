"""Prescriptive optimization: allocate retention offers using OR-Tools CP-SAT."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from ortools.sat.python import cp_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "offer_plan.csv"

# Offer tiers: (name, cost, uplift)
OFFER_TIERS = [
    ("none",       0,  0.00),
    ("5_pct",      2,  0.05),
    ("10_pct",     5,  0.10),
    ("20_pct",    10,  0.18),
]

SCALE = 100  # scale floats to integers for CP-SAT


def optimize(
    df: pd.DataFrame,
    budget: float = 5000.0,
    max_large_pct: float = 0.10,
) -> pd.DataFrame:
    """Solve the offer-allocation problem.

    Maximize: sum over customers of (monetary_90d * p_churn * uplift - cost)
    Subject to:
      - total cost <= budget
      - number of 20% coupons <= max_large_pct * n_customers
      - each customer gets exactly one offer tier
    """
    n = len(df)
    n_tiers = len(OFFER_TIERS)
    max_large = int(max_large_pct * n)

    monetary = df["monetary_90d"].fillna(0).values
    p_churn = df["p_churn"].fillna(0).values

    # Precompute scaled net benefit per (customer, tier)
    benefits = np.zeros((n, n_tiers), dtype=int)
    costs = []
    for t, (_, cost, uplift) in enumerate(OFFER_TIERS):
        net = monetary * p_churn * uplift - cost
        benefits[:, t] = (net * SCALE).astype(int)
        costs.append(cost)

    print(f"Customers: {n:,}  |  Budget: ${budget:,.0f}  |  Max 20% coupons: {max_large:,}")
    print("Building CP-SAT model ...")

    model = cp_model.CpModel()

    # Decision variables: x[i][t] = 1 if customer i gets tier t
    x = {}
    for i in range(n):
        for t in range(n_tiers):
            x[i, t] = model.new_bool_var(f"x_{i}_{t}")

    # Each customer gets exactly one tier
    for i in range(n):
        model.add_exactly_one(x[i, t] for t in range(n_tiers))

    # Budget constraint
    model.add(
        sum(int(costs[t] * SCALE) * x[i, t] for i in range(n) for t in range(n_tiers))
        <= int(budget * SCALE)
    )

    # Large coupon cap (tier index 3 = 20%)
    model.add(sum(x[i, 3] for i in range(n)) <= max_large)

    # Objective: maximize total net benefit
    model.maximize(
        sum(benefits[i, t] * x[i, t] for i in range(n) for t in range(n_tiers))
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120
    solver.parameters.num_workers = 4
    print("Solving ...")
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Solver failed with status {solver.status_name(status)}")

    # Extract solution
    tier_names = [t[0] for t in OFFER_TIERS]
    chosen_tier = []
    chosen_cost = []
    chosen_uplift = []
    for i in range(n):
        for t in range(n_tiers):
            if solver.value(x[i, t]):
                chosen_tier.append(tier_names[t])
                chosen_cost.append(costs[t])
                chosen_uplift.append(OFFER_TIERS[t][2])
                break

    df = df.copy()
    df["offer"] = chosen_tier
    df["offer_cost"] = chosen_cost
    df["offer_uplift"] = chosen_uplift
    df["expected_saved_revenue"] = df["monetary_90d"] * df["p_churn"] * df["offer_uplift"]

    # Summary
    total_cost = df["offer_cost"].sum()
    total_saved = df["expected_saved_revenue"].sum()
    print(f"\nSolver status : {solver.status_name(status)}")
    print(f"Total cost    : ${total_cost:,.2f}")
    print(f"Expected saved: ${total_saved:,.2f}")
    print(f"Net benefit   : ${total_saved - total_cost:,.2f}")
    print(f"\nOffer distribution:")
    print(df["offer"].value_counts().to_string())

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize retention offers")
    parser.add_argument("--budget", type=float, default=5000.0,
                        help="Total marketing budget (default: 5000)")
    parser.add_argument("--max-large-pct", type=float, default=0.10,
                        help="Max fraction of customers receiving 20%% coupon (default: 0.10)")
    args = parser.parse_args()

    print("Loading features ...")
    df = pd.read_parquet(FEATURES_PATH)

    if "p_churn" not in df.columns:
        raise ValueError("Column 'p_churn' not found. Run training first: python -m src.train")

    result = optimize(df, budget=args.budget, max_large_pct=args.max_large_pct)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nOffer plan saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
