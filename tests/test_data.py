import pandas as pd
import numpy as np
import pytest
from cwfc.data import build_aggregate_targets, get_train_holdout_split


def test_build_aggregate_targets():
    # 2 VMs, 3 timestamps (spacing 300)
    df_cpu = pd.DataFrame({
        "timestamp": [0, 0, 300, 300, 600, 600],
        "vm_id": ["vm1", "vm2", "vm1", "vm2", "vm1", "vm2"],
        "avg_cpu": [10.0, 20.0, 12.0, 18.0, 14.0, 16.0],
    })
    agg = build_aggregate_targets(df_cpu)
    assert len(agg) == 3
    assert list(agg["timestamp"]) == [0, 300, 600]
    assert list(agg["active_vm_count"]) == [2, 2, 2]
    assert list(agg["aggregate_cpu_utilization"]) == [15.0, 15.0, 15.0]


def test_get_train_holdout_split():
    df_agg = pd.DataFrame({
        "timestamp": range(0, 134 * 300, 300),
        "active_vm_count": [10] * 134,
        "aggregate_cpu_utilization": [5.0] * 134,
    })
    train_df, holdout_df = get_train_holdout_split(df_agg, train_size=110, holdout_size=24)
    assert len(train_df) == 110
    assert len(holdout_df) == 24
    assert train_df["timestamp"].iloc[-1] < holdout_df["timestamp"].iloc[0]


def test_get_train_holdout_split_smoke_test():
    # Smoke test with tiny dataset of length 10
    df_agg = pd.DataFrame({
        "timestamp": range(0, 10 * 300, 300),
        "active_vm_count": [5] * 10,
        "aggregate_cpu_utilization": [8.0] * 10,
    })
    train_df, holdout_df = get_train_holdout_split(df_agg, train_size=110, holdout_size=24)
    assert len(train_df) == 8
    assert len(holdout_df) == 2
    assert train_df["timestamp"].iloc[-1] < holdout_df["timestamp"].iloc[0]
