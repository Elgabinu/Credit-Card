"""
preprocess.py
=============
Loading and preprocessing transaction data.

Used by both training (model_training.py) and inference (prediction.py / app.py):

    from preprocess import load_transactions, prepare_features

`preprocessing.py` remains as a backward-compatible alias for older scripts.
"""

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────────
# V1-V28 are PCA components; Amount is the only raw numeric feature.
# Time is kept for display/ordering but excluded from the model to avoid leakage.
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"


# ─────────────────────────────────────────────
# 1. Loading
# ─────────────────────────────────────────────

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load the labelled creditcard.csv dataset used for training."""
    df = pd.read_csv(filepath)

    required = ["Time"] + FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"[load_data] Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


def load_transactions(source) -> pd.DataFrame:
    """
    Read an inference CSV from a path or a file-like object.

    Streamlit's `UploadedFile` is a file-like object, so the same function works
    for both the UI and the command line.
    """
    df = pd.read_csv(source)
    validate_features(df)
    return df


def validate_features(df: pd.DataFrame, require_target: bool = False) -> None:
    """Raise a helpful error if required feature (or target) columns are absent."""
    required = list(FEATURE_COLS)
    if require_target:
        required.append(TARGET_COL)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the model features, in the exact order used at training time."""
    validate_features(df)
    return df[FEATURE_COLS].copy()


# ─────────────────────────────────────────────
# 2. Chronological train / validation / test split
# ─────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by Time and split chronologically: 60 / 20 / 20.

    No shuffling - preserving temporal ordering prevents future information
    leaking into the training set.
    """
    df_sorted = df.sort_values("Time").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        fraud_pct = split[TARGET_COL].mean() * 100
        print(
            f"[split_data] {name:5s}: {len(split):,} rows | "
            f"Fraud: {int(split[TARGET_COL].sum()):,} ({fraud_pct:.3f}%)"
        )

    return train, val, test


# ─────────────────────────────────────────────
# 3. Feature / target separation
# ─────────────────────────────────────────────

def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix X and target vector y."""
    validate_features(df, require_target=True)
    return df[FEATURE_COLS].copy(), df[TARGET_COL].copy()


# ─────────────────────────────────────────────
# 4. Preprocessing pipeline
# ─────────────────────────────────────────────

def build_preprocessor() -> Pipeline:
    """
    Build the preprocessing step.

    V1-V28 are already PCA-scaled; Amount is scaled here. Scaling everything
    uniformly keeps the pipeline simple and matches the training artifact.
    """
    return Pipeline(steps=[("scaler", StandardScaler())])


# ─────────────────────────────────────────────
# 5. Class-imbalance helper
# ─────────────────────────────────────────────

def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """negatives / positives, for XGBoost's scale_pos_weight."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError("Training split contains no positive (fraud) examples.")

    ratio = n_neg / n_pos
    print(f"[scale_pos_weight] neg={n_neg:,} / pos={n_pos:,} -> ratio={ratio:.2f}")
    return ratio
