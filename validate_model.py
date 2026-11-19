"""
validate_model.py
=================
Sanity-check the saved model artifact.

Two modes:

1. Score a CSV that contains the ground-truth `Class` column (default:
   `sample_with_fraud.csv`) and print precision / recall / PR-AUC.

       python validate_model.py

2. Reproduce the held-out test-set metrics from `creditcard.csv`,
   using exactly the same chronological 60/20/20 split as training:

       python validate_model.py --test-split

This is useful after copying `models/best_pipeline.pkl` to a new machine,
because scikit-learn warns when an artifact was saved with a different
library version.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocessing import FEATURE_COLS, get_X_y, load_data, split_data

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "best_pipeline.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"


def load_artifacts():
    if not PIPELINE_PATH.exists():
        sys.exit(f"[ERROR] Model not found: {PIPELINE_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)

    threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = json.load(f)["threshold"]

    return pipeline, threshold


def report(y_true, proba, threshold, title):
    preds = (proba >= threshold).astype(int)
    print(f"\n=== {title} ===")
    print(f"  transactions : {len(y_true):,}")
    print(f"  fraud cases  : {int(y_true.sum()):,}")
    print(f"  threshold    : {threshold:.4f}")
    print(f"  PR-AUC       : {average_precision_score(y_true, proba):.4f}")
    print(f"  ROC-AUC      : {roc_auc_score(y_true, proba):.4f}")
    print(f"  precision    : {precision_score(y_true, preds, zero_division=0):.4f}")
    print(f"  recall       : {recall_score(y_true, preds, zero_division=0):.4f}")
    print(f"  F1           : {f1_score(y_true, preds, zero_division=0):.4f}")
    print("  confusion matrix:")
    print(confusion_matrix(y_true, preds))


def main():
    parser = argparse.ArgumentParser(description="Validate the saved fraud model")
    parser.add_argument(
        "--input",
        default=str(ROOT / "sample_with_fraud.csv"),
        help="CSV with a 'Class' column (default: sample_with_fraud.csv)",
    )
    parser.add_argument(
        "--test-split",
        action="store_true",
        help="Reproduce metrics on the chronological test split of creditcard.csv",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    pipeline, threshold = load_artifacts()
    print(f"[validate] Pipeline: {type(pipeline).__name__} | threshold: {threshold:.4f}")

    if args.test_split:
        df = load_data(str(ROOT / "creditcard.csv"))
        df = df.drop_duplicates()
        _, _, test = split_data(df)
        X_test, y_test = get_X_y(test)
        report(y_test, pipeline.predict_proba(X_test)[:, 1], threshold, "Held-out test split")
        return

    df = pd.read_csv(args.input)
    if "Class" not in df.columns:
        sys.exit("[ERROR] Input needs a 'Class' column for validation.")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing columns: {missing}")

    proba = pipeline.predict_proba(df[FEATURE_COLS])[:, 1]
    report(df["Class"].values, proba, threshold, args.input)


if __name__ == "__main__":
    main()
