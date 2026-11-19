"""
preprocessing.py
================
Backward-compatible alias for :mod:`preprocess`.

The canonical preprocessing module is now ``preprocess.py`` (module 3.2 in the
project report). This shim keeps older scripts working:

    from preprocessing import FEATURE_COLS, get_X_y, load_data, split_data

New code should import from ``preprocess`` directly.
"""

from preprocess import (  # noqa: F401
    FEATURE_COLS,
    TARGET_COL,
    build_preprocessor,
    compute_scale_pos_weight,
    get_X_y,
    load_data,
    load_transactions,
    prepare_features,
    split_data,
    validate_features,
)

__all__ = [
    "FEATURE_COLS",
    "TARGET_COL",
    "build_preprocessor",
    "compute_scale_pos_weight",
    "get_X_y",
    "load_data",
    "load_transactions",
    "prepare_features",
    "split_data",
    "validate_features",
]
