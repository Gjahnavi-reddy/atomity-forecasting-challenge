"""Command-line interface to train model on training window and generate predictions.csv.

Usage:
    python -m cwfc.predict --data challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd

from cwfc.data import load_raw_data, build_aggregate_targets, get_train_holdout_split
from cwfc.models import WorkloadForecaster


def validate_submission_frame(df: pd.DataFrame, expected_timestamps: list[int | float]) -> None:
    """Validate that the predictions dataframe strictly complies with SUBMISSION.md."""
    expected_cols = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Columns mismatch: expected {expected_cols}, got {list(df.columns)}")

    if len(df) != len(expected_timestamps):
        raise ValueError(f"Expected exactly {len(expected_timestamps)} prediction rows, got {len(df)}")

    if list(df["timestamp"]) != expected_timestamps:
        raise ValueError("Timestamps do not match expected holdout timestamps in order")

    if df.isna().any().any():
        raise ValueError("Predictions contain null or missing values")

    for col in ["active_vm_count", "aggregate_cpu_utilization"]:
        if not np.all(np.isfinite(df[col].values)):
            raise ValueError(f"Column '{col}' contains non-finite values (inf or nan)")

    if not np.all(df["active_vm_count"].values >= 0):
        raise ValueError("active_vm_count contains negative values")

    cpu_vals = df["aggregate_cpu_utilization"].values
    if not np.all((cpu_vals >= 0.0) & (cpu_vals <= 100.0)):
        raise ValueError("aggregate_cpu_utilization contains values outside [0, 100]")


def generate_predictions(
    data_dir: str = "challenge_data",
    output_path: str = "predictions.csv",
    model_type: str = "ridge",
    train_size: int = 110,
    holdout_size: int = 24,
) -> pd.DataFrame:
    """Load data, build training aggregate series, fit forecaster, and write predictions.csv."""
    print(f"Loading challenge data from: {data_dir}")
    df_cpu, df_meta = load_raw_data(data_dir=data_dir)

    print("Constructing aggregate target series...")
    df_agg = build_aggregate_targets(df_cpu, df_meta=df_meta)

    print(f"Splitting into train ({train_size} obs) and holdout ({holdout_size} obs)...")
    train_df, holdout_df = get_train_holdout_split(df_agg, train_size=train_size, holdout_size=holdout_size)
    holdout_timestamps = holdout_df["timestamp"].tolist()

    print(f"Fitting WorkloadForecaster (model_type='{model_type}')...")
    forecaster = WorkloadForecaster(cpu_model_type=model_type, alpha=1.0, horizon=holdout_size)
    forecaster.fit(train_df)

    print(f"Generating forecasts for the 2-hour ({holdout_size} steps) holdout horizon...")
    pred_df = forecaster.forecast(train_df, holdout_timestamps)

    print("Validating submission constraints...")
    validate_submission_frame(pred_df, holdout_timestamps)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pred_df.to_csv(output_path, index=False)
    print(f"Successfully generated compliant predictions at: {output_path}")
    print(pred_df.head(5))
    print(f"... ({len(pred_df)} rows total)")

    return pred_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for Cloud Workload Forecasting Challenge."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="challenge_data",
        help="Path to directory containing dataset files (default: challenge_data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to output predictions.csv (default: predictions.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mean_reverting",
        choices=["mean_reverting", "ridge", "gbm", "persistence", "moving_average"],
        help="Forecasting model to use (default: mean_reverting)",
    )

    args = parser.parse_args()
    try:
        generate_predictions(
            data_dir=args.data,
            output_path=args.output,
            model_type=args.model,
        )
    except Exception as e:
        print(f"Error during prediction generation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
