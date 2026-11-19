"""
visualization.py
================
Fraud statistics and probability distribution helpers.

Usage:
    from visualization import fraud_summary, probability_distribution

    summary = fraud_summary(preds)
    dist = probability_distribution(proba)
    st.bar_chart(dist)
"""

import numpy as np
import pandas as pd


def fraud_summary(preds) -> dict:
    """
    Headline counts for the results header.

    Returns
    -------
    dict
        total, fraud, legit, fraud_rate (percentage).
    """
    preds = np.asarray(preds)
    total = int(len(preds))
    fraud = int(preds.sum())
    return {
        "total": total,
        "fraud": fraud,
        "legit": total - fraud,
        "fraud_rate": (fraud / total * 100) if total else 0.0,
    }


def probability_distribution(proba, bins: int = 20) -> pd.DataFrame:
    """
    Bin probabilities into `bins` equal-width ranges for a bar chart.

    Returns a DataFrame indexed by readable text ranges ("0.00–0.05") with a
    single `transactions` column. Text labels avoid Altair treating the bins
    as a continuous (unbinned) scale.
    """
    proba = np.asarray(proba, dtype=float)

    # Sightly expand the top edge so p == 1.0 is included.
    edges = np.linspace(0.0, 1.0, bins + 1)
    edges[-1] += 1e-9

    counts = pd.cut(proba, bins=edges, include_lowest=True).value_counts().sort_index()

    labels = [
        f"{max(iv.left, 0.0):.2f}–{min(iv.right, 1.0):.2f}" for iv in counts.index
    ]

    return pd.DataFrame(
        {"probability range": labels, "transactions": counts.to_numpy()}
    ).set_index("probability range")


def highlight_fraud(row, fraud_label: str = "🚨 FRAUD"):
    """Pandas Styler row function: tint fraud rows red."""
    colour = "#ffcccc" if row.get("Prediction") == fraud_label else ""
    return [f"background-color: {colour}"] * len(row)
