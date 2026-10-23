"""Time-aware validation utilities for cloud workload time-series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np

from cwfc.metrics import mae, rmse, smape, normalized_mae


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")

    raw_ts = frame[timestamp_col]
    if pd.api.types.is_numeric_dtype(raw_ts):
        ts = pd.to_datetime(raw_ts, unit="s", utc=True)
    else:
        ts = pd.to_datetime(raw_ts, utc=True)

    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out


def evaluate_model_rolling_cv(
    forecaster: Any,
    train_df: pd.DataFrame,
    target_col: str = "aggregate_cpu_utilization",
    horizon: int = 24,
    min_train_size: int = 42,
    step: int = 11,
    baseline_forecaster: Any | None = None,
) -> dict[str, Any]:
    """Perform expanding-window temporal cross-validation on the training set.

    Args:
        forecaster: Model instance with `.fit(train_slice, target_col)` and
                    `.predict_horizon(history_slice, target_col, horizon)`.
        train_df: Chronological training DataFrame (first 110 observations).
        target_col: Target column to evaluate.
        horizon: Evaluation horizon (default: 24 steps = 2 hours).
        min_train_size: Minimum observations in initial train fold (default: 38).
        step: Step size between expanding origins (default: 12 steps = 1 hour).
        baseline_forecaster: Optional baseline model for normalized MAE computation.

    Returns:
        dict containing per-fold metrics, aggregate metrics, and step-by-step errors.
    """
    n = len(train_df)
    origins = list(range(min_train_size, n - horizon + 1, step))
    if not origins:
        raise ValueError(f"No valid origins found with n={n}, min_train_size={min_train_size}, horizon={horizon}")

    fold_results: list[dict[str, Any]] = []
    step_errors: list[np.ndarray] = []

    for fold_idx, origin in enumerate(origins):
        train_slice = train_df.iloc[:origin].copy().reset_index(drop=True)
        val_slice = train_df.iloc[origin : origin + horizon].copy().reset_index(drop=True)
        y_true = val_slice[target_col].values

        # Fit model strictly on train_slice
        forecaster.fit(train_slice, target_col)
        y_pred = forecaster.predict_horizon(train_slice, target_col, horizon=horizon)

        fold_mae = mae(y_true, y_pred)
        fold_rmse = rmse(y_true, y_pred)
        fold_smape = smape(y_true, y_pred)

        fold_dict = {
            "fold": fold_idx + 1,
            "origin_index": origin,
            "train_size": len(train_slice),
            "mae": fold_mae,
            "rmse": fold_rmse,
            "smape": fold_smape,
        }

        if baseline_forecaster is not None:
            baseline_forecaster.fit(train_slice, target_col)
            y_base = baseline_forecaster.predict_horizon(train_slice, target_col, horizon=horizon)
            fold_dict["baseline_mae"] = mae(y_true, y_base)
            fold_dict["normalized_mae"] = normalized_mae(y_true, y_pred, y_base)

        fold_results.append(fold_dict)
        step_errors.append(np.abs(y_true - y_pred))

    step_mae_by_horizon = np.mean(step_errors, axis=0)

    summary = {
        "num_folds": len(origins),
        "mean_mae": float(np.mean([f["mae"] for f in fold_results])),
        "std_mae": float(np.std([f["mae"] for f in fold_results])),
        "mean_rmse": float(np.mean([f["rmse"] for f in fold_results])),
        "std_rmse": float(np.std([f["rmse"] for f in fold_results])),
        "mean_smape": float(np.mean([f["smape"] for f in fold_results])),
        "std_smape": float(np.std([f["smape"] for f in fold_results])),
        "step_mae_by_horizon": step_mae_by_horizon.tolist(),
        "folds": fold_results,
    }

    if baseline_forecaster is not None:
        summary["mean_normalized_mae"] = float(np.mean([f["normalized_mae"] for f in fold_results]))

    return summary
