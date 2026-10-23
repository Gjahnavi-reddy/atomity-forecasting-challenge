import pandas as pd
import numpy as np
from cwfc.models import PersistenceForecaster, MovingAverageForecaster, DirectMultiHorizonForecaster, WorkloadForecaster


def test_forecasters_basic():
    df = pd.DataFrame({
        "timestamp": range(0, 50 * 300, 300),
        "active_vm_count": [1000] * 50,
        "aggregate_cpu_utilization": np.linspace(10, 20, 50),
    })

    # Persistence
    pers = PersistenceForecaster()
    pers.fit(df, "aggregate_cpu_utilization")
    p_pred = pers.predict_horizon(df, "aggregate_cpu_utilization", horizon=5)
    assert len(p_pred) == 5
    assert np.all(p_pred == 20.0)

    # Moving Average
    ma = MovingAverageForecaster(window=5)
    ma.fit(df, "aggregate_cpu_utilization")
    ma_pred = ma.predict_horizon(df, "aggregate_cpu_utilization", horizon=5)
    assert len(ma_pred) == 5
    assert np.isclose(ma_pred[0], np.mean(np.linspace(10, 20, 50)[-5:]))

    # Direct Multi Horizon Ridge
    ridge_forecaster = DirectMultiHorizonForecaster(model_type="ridge", horizon=5, min_history=10)
    ridge_forecaster.fit(df, "aggregate_cpu_utilization")
    r_pred = ridge_forecaster.predict_horizon(df, "aggregate_cpu_utilization", horizon=5)
    assert len(r_pred) == 5
    assert np.all(np.isfinite(r_pred))

    # WorkloadForecaster pipeline
    wf = WorkloadForecaster(cpu_model_type="ridge", horizon=5)
    wf.fit(df)
    holdout_ts = list(range(50 * 300, 55 * 300, 300))
    pred_df = wf.forecast(df, holdout_ts)
    assert list(pred_df.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(pred_df) == 5
    assert np.all(pred_df["active_vm_count"] >= 0)
    assert np.all(pred_df["aggregate_cpu_utilization"] >= 0)
