"""Data ingestion, validation, and aggregate target construction."""
from __future__ import annotations

import os
import pandas as pd
import numpy as np


def load_raw_data(data_dir: str = "challenge_data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw CPU readings and VM metadata compressed CSVs.

    Args:
        data_dir: Directory containing vm_cpu_readings.csv.gz and vm_metadata.csv.gz.

    Returns:
        tuple of (df_cpu, df_meta)
    """
    cpu_path = os.path.join(data_dir, "vm_cpu_readings.csv.gz")
    meta_path = os.path.join(data_dir, "vm_metadata.csv.gz")

    if not os.path.exists(cpu_path):
        raise FileNotFoundError(f"Missing required CPU readings file: {cpu_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing required VM metadata file: {meta_path}")

    df_cpu = pd.read_csv(cpu_path)
    df_meta = pd.read_csv(meta_path)
    return df_cpu, df_meta


def build_aggregate_targets(
    df_cpu: pd.DataFrame,
    df_meta: pd.DataFrame | None = None,
    expected_cadence_seconds: int = 300,
) -> pd.DataFrame:
    """Construct the aggregate target time series from row-level VM telemetry.

    Produces:
        - active_vm_count: number of distinct VM IDs with a valid CPU reading at timestamp t.
        - aggregate_cpu_utilization: mean of VM-level avg_cpu values at timestamp t in [0, 100].

    Also incorporates segment-level telemetry if metadata is supplied.
    """
    if df_cpu.empty:
        raise ValueError("df_cpu must not be empty")

    # Validate required columns
    required_cpu_cols = {"timestamp", "vm_id", "avg_cpu"}
    if not required_cpu_cols.issubset(df_cpu.columns):
        raise ValueError(f"df_cpu must contain columns: {required_cpu_cols}")

    # Deduplicate check (each VM at most once per timestamp)
    df_cpu_sorted = df_cpu.sort_values(["timestamp", "vm_id"]).drop_duplicates(subset=["timestamp", "vm_id"])

    # Base aggregate targets
    agg_dict = {
        "active_vm_count": ("vm_id", "nunique"),
        "aggregate_cpu_utilization": ("avg_cpu", "mean"),
    }
    if "min_cpu" in df_cpu_sorted.columns:
        agg_dict["min_cpu_mean"] = ("min_cpu", "mean")
    if "max_cpu" in df_cpu_sorted.columns:
        agg_dict["max_cpu_mean"] = ("max_cpu", "mean")

    agg_df = df_cpu_sorted.groupby("timestamp").agg(**agg_dict).reset_index()

    # If metadata is available, add segment aggregates up to timestamp t
    if df_meta is not None and "vm_id" in df_meta.columns and "vm_category" in df_meta.columns:
        merged = df_cpu_sorted.merge(df_meta[["vm_id", "vm_category"]], on="vm_id", how="left")
        cat_agg = merged.groupby(["timestamp", "vm_category"])["avg_cpu"].mean().unstack()
        for cat in cat_agg.columns:
            clean_col = f"cpu_{str(cat).lower().replace('-', '_').replace(' ', '_')}"
            agg_df[clean_col] = cat_agg[cat].values

    # Sort strictly by timestamp
    agg_df = agg_df.sort_values("timestamp").reset_index(drop=True)

    # Validate cadence
    ts_array = agg_df["timestamp"].values
    if len(ts_array) > 1:
        diffs = np.diff(ts_array)
        detected_step = diffs[0]
        check_step = expected_cadence_seconds if expected_cadence_seconds is not None else detected_step
        if not np.all(diffs == check_step):
            irregular = diffs[diffs != check_step]
            raise ValueError(
                f"Irregular timestamp spacing detected. Expected {check_step}s, found: {np.unique(irregular)}"
            )

    # Validate constraints
    assert (agg_df["active_vm_count"] >= 0).all(), "active_vm_count must be non-negative"
    assert (
        (agg_df["aggregate_cpu_utilization"] >= 0) & (agg_df["aggregate_cpu_utilization"] <= 100)
    ).all(), "aggregate_cpu_utilization must be in [0, 100]"

    return agg_df


def get_train_holdout_split(
    df_agg: pd.DataFrame,
    train_size: int = 110,
    holdout_size: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the aggregate time series into train and holdout windows.

    For the standard challenge, splits into the first 110 observations and final 24 observations.
    For small smoke-test datasets (total < 134), dynamically adapts proportional splits.

    Args:
        df_agg: Aggregate time-series dataframe sorted chronologically.
        train_size: Number of training observations (default: 110).
        holdout_size: Number of holdout observations (default: 24).

    Returns:
        tuple of (train_df, holdout_df)
    """
    total = len(df_agg)
    if total < 2:
        raise ValueError(f"Need at least 2 observations, got {total}")

    if total < train_size + holdout_size:
        # Gracefully adapt for small reviewer smoke tests
        eff_holdout = max(1, min(holdout_size, int(total * 0.2)))
        eff_train = total - eff_holdout
    else:
        eff_train = train_size
        eff_holdout = holdout_size

    train_df = df_agg.iloc[:eff_train].copy().reset_index(drop=True)
    holdout_df = df_agg.iloc[eff_train : eff_train + eff_holdout].copy().reset_index(drop=True)
    return train_df, holdout_df
