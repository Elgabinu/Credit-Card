"""
prediction.py
=============
Fraud probability prediction and classification.

Turns raw transactions into:

    fraud_probability  -> model score in [0, 1]
    Prediction         -> "✅ Legitimate" or "🚨 FRAUD"

Usage:
    from prediction import load_artifacts, score_transactions

    pipeline, threshold = load_artifacts()
    df_out, proba, preds = score_transactions(df, pipeline, threshold)
"""

from pathlib import Path

import joblib
import pandas as pd

from preprocess import FEATURE_COLS

MODEL_DIR = Path(__file__).parent / "models"


def load_artifacts(model_dir: str | Path = MODEL_DIR):
    """
    Load the trained pipeline and its decision threshold.

    Returns
    -------
    (pipeline, threshold)
        `threshold` falls back to 0.5 when `threshold.json` is absent.
    """
    model_dir = Path(model_dir)
    pipeline_path = model_dir / "best_pipeline.pkl"
    threshold_path = model_dir / "threshold.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model not found: {pipeline_path}")

    pipeline = joblib.load(pipeline_path)

    threshold = 0.5
    if threshold_path.exists():
        import json

        with open(threshold_path) as f:
            threshold = float(json.load(f)["threshold"])

    return pipeline, threshold


def predict_proba(df: pd.DataFrame, pipeline) -> "pd.Series":
    """Return the fraud probability for every row."""
    return pipeline.predict_proba(df[FEATURE_COLS])[:, 1]


def classify(proba, threshold: float = 0.5):
    """Convert probabilities into 0/1 predictions at the given threshold."""
    import numpy as np

    return (np.asarray(proba) >= threshold).astype(int)


def score_transactions(df: pd.DataFrame, pipeline, threshold: float = 0.5):
    """
    Score a transaction DataFrame.

    Returns
    -------
    df_out : DataFrame
        Copy of the input with `fraud_probability` and `Prediction` columns.
    proba : ndarray
        Fraud probability per row.
    preds : ndarray
        0/1 prediction per row.
    """
    proba = predict_proba(df, pipeline)
    preds = classify(proba, threshold)

    df_out = df.copy()
    df_out["fraud_probability"] = proba.round(4)
    df_out["Prediction"] = pd.Series(preds).map({0: "✅ Legitimate", 1: "🚨 FRAUD"}).values

    return df_out, proba, preds
