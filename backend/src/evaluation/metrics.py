"""
Evaluation metrics - shared by training pipelines and evaluation scripts.

Keeps metric computation in one place so the train pipeline, walk-forward
validation, and future model-comparison pages all report identical numbers.
"""
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def classification_metrics(y_true, y_pred, threshold: float = 0.5) -> Dict[str, float]:
    """Binary up/down metrics: accuracy, balanced accuracy, P/R/F1, confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def regression_metrics(y_true_reg, y_pred_reg) -> Dict[str, float]:
    """Continuous-target metrics: MSE, MAE, R2."""
    return {
        "mse": float(mean_squared_error(y_true_reg, y_pred_reg)),
        "mae": float(mean_absolute_error(y_true_reg, y_pred_reg)),
        "r2": float(r2_score(y_true_reg, y_pred_reg)),
    }
