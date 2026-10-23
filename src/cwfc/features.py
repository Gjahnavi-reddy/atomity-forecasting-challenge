"""Feature engineering for time-series cloud workload forecasting.

All features are strictly leakage-free: at forecast origin t, features are derived
only from observations at or prior to t (t' <= t).
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def extract_origin_features(
    history_df: pd.DataFrame,
    target_col: str = "aggregate_cpu_utilization",
    lags: tuple[int, ...] = (0, 1, 2, 3, 5, 11),
    windows: tuple[int, ...] = (3, 6, 12),
) -> dict[str, float]:
    """Compute a 1-D feature dictionary at a single forecast origin t.

    `history_df` must contain observations strictly up to and including the forecast origin t.
    The last row of `history_df` is assumed to be the observation at origin t.
    """
    if history_df.empty:
        raise ValueError("history_df must not be empty")

    values = history_df[target_col].astype(float).values
    n = len(values)
    current_val = values[-1]
    features: dict[str, float] = {}

    # 1. Target lags
    for lag in lags:
        idx = -1 - lag
        if abs(idx) <= n:
            features[f"{target_col}_lag_{lag}"] = float(values[idx])
        else:
            features[f"{target_col}_lag_{lag}"] = float(values[0])

    # 2. Rolling window statistics
    for w in windows:
        w_eff = min(w, n)
        window_vals = values[-w_eff:]
        features[f"{target_col}_roll_mean_{w}"] = float(np.mean(window_vals))
        features[f"{target_col}_roll_std_{w}"] = float(np.std(window_vals, ddof=1)) if len(window_vals) > 1 else 0.0
        features[f"{target_col}_roll_min_{w}"] = float(np.min(window_vals))
        features[f"{target_col}_roll_max_{w}"] = float(np.max(window_vals))
        features[f"{target_col}_roll_range_{w}"] = float(np.max(window_vals) - np.min(window_vals))

    # 3. Differencing / Momentum
    if n >= 2:
        features[f"{target_col}_diff_1"] = float(current_val - values[-2])
    else:
        features[f"{target_col}_diff_1"] = 0.0

    if n >= 4:
        features[f"{target_col}_diff_3"] = float(current_val - values[-4])
    else:
        features[f"{target_col}_diff_3"] = 0.0

    roll_mean_6 = features.get(f"{target_col}_roll_mean_6", current_val)
    features[f"{target_col}_diff_from_roll_mean_6"] = float(current_val - roll_mean_6)

    # 4. Optional segment features if present in history_df
    for seg_col in ["cpu_interactive", "cpu_delay_insensitive", "cpu_unknown"]:
        if seg_col in history_df.columns:
            seg_vals = history_df[seg_col].astype(float).values
            features[f"{seg_col}_current"] = float(seg_vals[-1])
            if len(seg_vals) >= 3:
                features[f"{seg_col}_roll_mean_3"] = float(np.mean(seg_vals[-3:]))
                features[f"{seg_col}_diff_1"] = float(seg_vals[-1] - seg_vals[-2])

    # 5. Elapsed time / trend index
    last_ts = float(history_df["timestamp"].iloc[-1])
    features["elapsed_hours"] = last_ts / 3600.0

    return features


def build_direct_forecasting_matrices(
    df: pd.DataFrame,
    target_col: str = "aggregate_cpu_utilization",
    horizon: int = 24,
    min_history: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct (X, Y) training matrices for direct multi-horizon forecasting.

    Each row i corresponds to a historical origin t_i.
    X[i] contains features computed strictly from observations t <= t_i.
    Y[i, h] contains the true target value at t_i + h (where h = 1..horizon).
    """
    n = len(df)
    if n < min_history + horizon:
        raise ValueError(f"Need at least {min_history + horizon} points, got {n}")

    x_rows = []
    y_rows = []

    for origin_idx in range(min_history, n - horizon + 1):
        history_slice = df.iloc[:origin_idx]
        future_slice = df.iloc[origin_idx : origin_idx + horizon]

        feat_dict = extract_origin_features(history_slice, target_col=target_col)
        x_rows.append(feat_dict)

        target_dict = {f"h_{h}": float(future_slice[target_col].iloc[h - 1]) for h in range(1, horizon + 1)}
        y_rows.append(target_dict)

    x_df = pd.DataFrame(x_rows)
    y_df = pd.DataFrame(y_rows)
    return x_df, y_df
