import pandas as pd
import numpy as np
from cwfc.features import extract_origin_features, build_direct_forecasting_matrices


def test_extract_origin_features():
    df = pd.DataFrame({
        "timestamp": range(0, 20 * 300, 300),
        "aggregate_cpu_utilization": np.linspace(10, 20, 20),
    })
    feats = extract_origin_features(df, target_col="aggregate_cpu_utilization")
    assert "aggregate_cpu_utilization_lag_0" in feats
    assert "aggregate_cpu_utilization_roll_mean_3" in feats
    assert feats["aggregate_cpu_utilization_lag_0"] == 20.0
    assert "elapsed_hours" in feats


def test_build_direct_forecasting_matrices():
    df = pd.DataFrame({
        "timestamp": range(0, 50 * 300, 300),
        "aggregate_cpu_utilization": np.random.uniform(5, 15, 50),
    })
    X, Y = build_direct_forecasting_matrices(
        df, target_col="aggregate_cpu_utilization", horizon=5, min_history=10
    )
    assert len(X) == len(Y)
    assert "h_1" in Y.columns
    assert "h_5" in Y.columns
    assert len(X) == 50 - 5 - 10 + 1
