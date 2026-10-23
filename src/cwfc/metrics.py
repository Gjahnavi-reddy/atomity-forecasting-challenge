from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    model_mae = mae(y_true, y_pred)
    baseline_mae = mae(y_true, baseline_pred)
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / baseline_mae)


def max_error(y_true, y_pred) -> float:
    """Return maximum absolute error across all predictions."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.max(np.abs(a - b)))


def peak_mae(y_true, y_pred, percentile: float = 90.0) -> float:
    """Return MAE restricted to the top high-load observations (>= given percentile of y_true)."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    threshold = np.percentile(a, percentile)
    peak_mask = a >= threshold
    if not np.any(peak_mask):
        return mae(a, b)
    return float(np.mean(np.abs(a[peak_mask] - b[peak_mask])))


def evaluate_metrics_summary(y_true, y_pred, baseline_pred=None) -> dict[str, float]:
    """Calculate a standard dictionary of all core evaluation metrics."""
    res = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "peak_mae": peak_mae(y_true, y_pred, percentile=90.0),
    }
    if baseline_pred is not None:
        res["normalized_mae"] = normalized_mae(y_true, y_pred, baseline_pred)
    return res
