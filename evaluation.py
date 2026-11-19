"""
evaluation.py
=============
Model evaluation: accuracy, precision, recall, F1-score and confusion matrix.

Usage:
    from evaluation import evaluate_predictions, confusion_matrix_frame

    metrics = evaluate_predictions(y_true, proba, preds)
    cm_df = confusion_matrix_frame(y_true, preds)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Row / column labels used by the UI.
CM_INDEX = ["Actual: Legit", "Actual: Fraud"]
CM_COLUMNS = ["Predicted: Legit", "Predicted: Fraud"]


def evaluate_predictions(y_true, proba=None, preds=None, threshold: float = 0.5) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (0 = legitimate, 1 = fraud).
    proba : array-like, optional
        Fraud probabilities. Required for ROC-AUC / PR-AUC.
    preds : array-like, optional
        0/1 predictions. Derived from `proba` and `threshold` when omitted.
    threshold : float
        Decision threshold used when `preds` is not supplied.

    Returns
    -------
    dict
        accuracy, precision, recall, f1, and (when computable) roc_auc / pr_auc.
    """
    y_true = np.asarray(y_true)
    if preds is None:
        if proba is None:
            raise ValueError("Provide either `preds` or `proba`.")
        preds = (np.asarray(proba) >= threshold).astype(int)
    preds = np.asarray(preds)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
    }

    # AUC is undefined when only one class is present.
    if proba is not None and len(np.unique(y_true)) > 1:
        proba = np.asarray(proba)
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, proba))

    return metrics


def confusion_matrix_frame(y_true, preds) -> pd.DataFrame:
    """
    Confusion matrix as a labelled DataFrame.

    Fixed `labels=[0, 1]` guarantees a 2x2 matrix even when one class is
    missing from the file.
    """
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    return pd.DataFrame(cm, index=CM_INDEX, columns=CM_COLUMNS)


def confusion_matrix_array(y_true, preds):
    """Raw 2x2 confusion matrix (NumPy array)."""
    return confusion_matrix(y_true, preds, labels=[0, 1])
