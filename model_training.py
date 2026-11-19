"""
model_training.py
=================
Training the Machine Learning model.

Fits a StandardScaler + tuned XGBoost pipeline on the chronological training
split, picks the best decision threshold on the validation split, and reports
held-out test metrics.

Usage:
    python model_training.py
    python model_training.py --data creditcard.csv --model-dir models

Artifacts written to the model directory:
    best_pipeline.pkl        trained scikit-learn Pipeline
    threshold.json           optimised decision threshold
    evaluation_results.json  test metrics and settings
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluation import evaluate_predictions
from preprocess import (
    compute_scale_pos_weight,
    get_X_y,
    load_data,
    split_data,
)

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "creditcard.csv"
DEFAULT_MODEL_DIR = ROOT / "models"

# Hyperparameters selected by RandomizedSearchCV in the Kaggle notebook.
TUNED_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.8,
)


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """StandardScaler followed by the tuned XGBoost classifier."""
    clf = XGBClassifier(
        **TUNED_PARAMS,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def find_best_threshold(y_true, proba) -> tuple[float, float]:
    """
    Pick the threshold that maximises F1 on the validation split.

    Returns
    -------
    (threshold, best_f1)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    # precision/recall have one more element than thresholds; drop the tail.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))

    return float(thresholds[best_idx]), float(f1[best_idx])


def train(
    data_path: str | Path = DEFAULT_DATA,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    drop_duplicates: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Train the full pipeline and persist all artifacts.

    Returns the evaluation results dictionary.
    """
    data_path = Path(data_path)
    model_dir = Path(model_dir)

    print("1. Loading dataset...")
    df = load_data(data_path)

    if drop_duplicates:
        n_dup = int(df.duplicated().sum())
        if n_dup:
            df = df.drop_duplicates()
            print(f"   Removed {n_dup:,} duplicate rows.")

    print("2. Chronological 60/20/20 split...")
    train_df, val_df, test_df = split_data(df)

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)

    print("3. Computing scale_pos_weight...")
    scale_pos_weight = compute_scale_pos_weight(y_train)

    print("4. Fitting pipeline...")
    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)

    print("5. Optimising decision threshold on validation split...")
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold, val_f1 = find_best_threshold(y_val, val_proba)
    print(f"   threshold={threshold:.10f} | validation F1={val_f1:.4f}")

    print("6. Evaluating on held-out test split...")
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)
    test_metrics = evaluate_predictions(
        y_test, proba=test_proba, preds=test_preds, threshold=threshold
    )

    if verbose:
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            if name in test_metrics:
                print(f"   {name:10s}: {test_metrics[name]:.4f}")

    results = {
        "model": "XGBoost",
        "threshold": threshold,
        "validation_f1": val_f1,
        "test": test_metrics,
        "settings": {
            "rows_after_dedup": int(len(df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "scale_pos_weight": float(scale_pos_weight),
            "hyperparameters": TUNED_PARAMS,
        },
    }

    print("7. Saving artifacts...")
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_dir / "best_pipeline.pkl")
    with open(model_dir / "threshold.json", "w") as f:
        json.dump({"threshold": threshold, "model": "XGBoost"}, f, indent=2)
    with open(model_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"   Saved to {model_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train the fraud detection model")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not drop duplicate rows before splitting.",
    )
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_dir=args.model_dir,
        drop_duplicates=not args.keep_duplicates,
    )


if __name__ == "__main__":
    main()
