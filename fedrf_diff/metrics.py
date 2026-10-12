"""Validation metrics computed from confusion counts.

Clients report only their confusion counts, so the Global Server row is computed from the
summed counts without any validation patient leaving its hospital.
"""

import math

import numpy as np

MCC = "Matthews Correlation Coefficient (MCC)"
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", MCC, "Cohen's Kappa"]


def confusion_counts(y_true, y_pred):
    """Return [tn, fp, fn, tp] for the positive class 1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.array([np.sum((y_true == 0) & (y_pred == 0)), np.sum((y_true == 0) & (y_pred == 1)),
                     np.sum((y_true == 1) & (y_pred == 0)), np.sum((y_true == 1) & (y_pred == 1))], dtype=np.int64)


def _ratio(a, b):
    return a / b if b else 0.0


def metrics_from_counts(counts):
    tn, fp, fn, tp = (int(v) for v in counts)
    n = tn + fp + fn + tp
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    p_obs = _ratio(tp + tn, n)
    p_exp = _ratio((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), n * n)
    return {
        "Accuracy": p_obs,
        "Precision": precision,
        "Recall": recall,
        "Specificity": _ratio(tn, tn + fp),
        "F1 Score": _ratio(2 * precision * recall, precision + recall),
        MCC: _ratio(tp * tn - fp * fn, mcc_den),
        "Cohen's Kappa": _ratio(p_obs - p_exp, 1 - p_exp),
    }
