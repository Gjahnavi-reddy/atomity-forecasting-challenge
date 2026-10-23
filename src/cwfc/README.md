# cwfc

Core Python utilities and forecasting models for the Cloud Workload Forecasting Challenge.

## Modules

- `data.py`: Loads raw dataset files, aggregates readings into `active_vm_count` and `aggregate_cpu_utilization`, and splits the 110/24 train and holdout windows.
- `features.py`: Computes lag, rolling window, and trend features at each forecast origin without lookahead.
- `models.py`: Forecasters including persistence, moving average, historical mean, direct multi-horizon models (Ridge, GBDT), and the mean-reverting AR model.
- `validation.py`: Rolling-origin temporal cross-validation (5 folds, horizon = 24 steps).
- `metrics.py`: Metrics covering MAE, RMSE, SMAPE, normalized MAE, and peak-period MAE.
- `predict.py`: CLI driver that trains the model on the training window and outputs `predictions.csv`.

## Quick usage

```python
from cwfc.data import load_raw_data, build_aggregate_targets, get_train_holdout_split
from cwfc.models import WorkloadForecaster

# Load and build 5-minute aggregate series
df_cpu, df_meta = load_raw_data("challenge_data")
df_agg = build_aggregate_targets(df_cpu, df_meta=df_meta)
train, holdout = get_train_holdout_split(df_agg, train_size=110, holdout_size=24)

# Fit model and predict the 2-hour holdout
forecaster = WorkloadForecaster(cpu_model_type="mean_reverting")
forecaster.fit(train)
preds = forecaster.forecast(train, holdout["timestamp"].tolist())
```
