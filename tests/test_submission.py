import os
import pandas as pd
import numpy as np
import pytest

from cwfc.predict import validate_submission_frame


def test_predictions_csv_contract():
    csv_path = "predictions.csv"
    assert os.path.exists(csv_path), "predictions.csv does not exist"

    df = pd.read_csv(csv_path)
    expected_cols = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert list(df.columns) == expected_cols

    assert len(df) == 24, f"Expected 24 rows, got {len(df)}"

    expected_timestamps = list(range(33000, 40200, 300))
    assert list(df["timestamp"]) == expected_timestamps

    assert not df.isna().any().any(), "predictions.csv contains nulls"

    assert np.all(np.isfinite(df["active_vm_count"].values)), "active_vm_count contains non-finite numbers"
    assert np.all(np.isfinite(df["aggregate_cpu_utilization"].values)), "aggregate_cpu_utilization contains non-finite numbers"

    assert np.all(df["active_vm_count"].values >= 0), "active_vm_count has negative values"
    assert np.all(
        (df["aggregate_cpu_utilization"].values >= 0.0) & (df["aggregate_cpu_utilization"].values <= 100.0)
    ), "aggregate_cpu_utilization outside [0, 100]"

    # Test the validation function itself
    validate_submission_frame(df, expected_timestamps)
