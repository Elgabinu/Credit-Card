"""
predict.py
==========
CLI script for local inference using the trained pipeline artifact.

Usage:
    python predict.py --input sample_transactions.csv --output predictions.csv

The input CSV must contain the same columns as creditcard.csv
(Time, V1-V28, Amount). The 'Class' column is optional.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import FEATURE_COLS


MODEL_DIR  = Path(__file__).parent / "models"
PIPELINE_PATH   = MODEL_DIR / "best_pipeline.pkl"
THRESHOLD_PATH  = MODEL_DIR / "threshold.json"


def load_artifacts():
    if not PIPELINE_PATH.exists():
        print(
            "[ERROR] Model file not found at models/best_pipeline.pkl\n"
            "Please download the trained model from your Kaggle notebook output\n"
            "and place it in the 'models/' folder."
        )
        sys.exit(1)

    pipeline = joblib.load(PIPELINE_PATH)

    threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = json.load(f)["threshold"]
        print(f"[predict] Using optimised threshold: {threshold:.4f}")
    else:
        print("[predict] threshold.json not found, defaulting to 0.5")

    return pipeline, threshold


def predict(input_path: str, output_path: str):
    print(f"[predict] Reading input: {input_path}")
    df = pd.read_csv(input_path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] Input CSV is missing columns: {missing}")
        sys.exit(1)

    X = df[FEATURE_COLS]

    pipeline, threshold = load_artifacts()

    proba = pipeline.predict_proba(X)[:, 1]
    predictions = (proba >= threshold).astype(int)

    df_out = df.copy()
    df_out["fraud_probability"] = proba.round(4)
    df_out["predicted_class"]   = predictions
    df_out["prediction_label"]  = pd.Series(predictions).map({0: "Legitimate", 1: "FRAUD"}).values

    df_out.to_csv(output_path, index=False)

    n_fraud = predictions.sum()
    print(f"\n[predict] Results:")
    print(f"  Total transactions : {len(predictions):,}")
    print(f"  Flagged as FRAUD   : {n_fraud:,}")
    print(f"  Fraud rate         : {n_fraud/len(predictions)*100:.3f}%")
    print(f"  Output saved to    : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection — Inference")
    parser.add_argument("--input",  required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="predictions.csv", help="Path to save predictions")
    args = parser.parse_args()
    predict(args.input, args.output)
