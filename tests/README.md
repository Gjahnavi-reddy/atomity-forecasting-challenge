# Tests

Unit tests covering data aggregation, feature engineering, models, submission formatting, and temporal splits.

## Running tests

```bash
pytest -v
```

## Coverage

- `test_data.py`: Tests aggregate target computation and train/holdout splitting (including smoke tests).
- `test_features.py`: Tests leakage-free feature extraction and multi-horizon matrix construction.
- `test_models.py`: Tests persistence, moving average, and machine learning models.
- `test_validation.py`: Tests monotonic expanding-window temporal splits.
- `test_metrics.py`: Tests MAE, RMSE, SMAPE, and normalized MAE.
- `test_submission.py`: Validates `predictions.csv` row count, column names, finite values, and bounds.
