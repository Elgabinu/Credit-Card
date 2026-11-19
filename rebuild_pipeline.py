"""
rebuild_pipeline.py
===================
Re-fit the tuned XGBoost pipeline locally.

Why this exists: `models/best_pipeline.pkl` produced by the Kaggle notebook was
serialised with the notebook's scikit-learn / XGBoost versions. Pickle files are
version-sensitive, so loading it in a different environment can fail with
`InconsistentVersionWarning` or `XGBoostError: input stream corrupted`.

This script refits the exact same pipeline (same split, same preprocessing,
same tuned hyperparameters found by RandomizedSearchCV on Kaggle) using the
library versions installed on *this* machine.

Usage:
    python rebuild_pipeline.py
    python rebuild_pipeline.py --data creditcard.csv --out models/best_pipeline.pkl

After rebuilding, confirm the metrics still match:

    python validate_model.py --test-split
"""

import argparse
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocessing import (
    compute_scale_pos_weight,
    get_X_y,
    load_data,
    split_data,
)

ROOT = Path(__file__).parent

# Hyperparameters selected by RandomizedSearchCV in the Kaggle notebook.
TUNED_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.8,
)


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    clf = XGBClassifier(
        **TUNED_PARAMS,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def main():
    parser = argparse.ArgumentParser(description="Rebuild the local model artifact")
    parser.add_argument("--data", default=str(ROOT / "creditcard.csv"))
    parser.add_argument("--out", default=str(ROOT / "models" / "best_pipeline.pkl"))
    args = parser.parse_args()

    print("1. Loading dataset...")
    df = load_data(args.data)

    n_dup = df.duplicated().sum()
    if n_dup:
        df = df.drop_duplicates()
        print(f"   Removed {n_dup} duplicate rows.")

    print("2. Chronological 60/20/20 split...")
    train_df, _, _ = split_data(df)
    X_train, y_train = get_X_y(train_df)

    print("3. Computing scale_pos_weight...")
    scale_pos_weight = compute_scale_pos_weight(y_train)

    print("4. Fitting pipeline...")
    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"5. Saved pipeline to {out_path}")


if __name__ == "__main__":
    main()
