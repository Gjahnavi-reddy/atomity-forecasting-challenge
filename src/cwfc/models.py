"""Forecasting models for cloud workload demand prediction.

Provides:
- BaseForecaster interface
- Simple baselines (Persistence, Moving Average, Historical Mean)
- DirectMultiHorizonForecaster (Ridge, GBDT, and Quantile Regressors)
- Unified WorkloadForecaster for both active_vm_count and aggregate_cpu_utilization
"""
from __future__ import annotations

from typing import Literal
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from cwfc.features import extract_origin_features, build_direct_forecasting_matrices


class BaseForecaster:
    """Base class for time-series forecasters."""

    def fit(self, train_df: pd.DataFrame, target_col: str) -> BaseForecaster:
        raise NotImplementedError

    def predict_horizon(self, history_df: pd.DataFrame, target_col: str, horizon: int = 24) -> np.ndarray:
        raise NotImplementedError


class PersistenceForecaster(BaseForecaster):
    """Forecasts the last observed value across all horizon steps."""

    def fit(self, train_df: pd.DataFrame, target_col: str) -> PersistenceForecaster:
        self.target_col = target_col
        return self

    def predict_horizon(self, history_df: pd.DataFrame, target_col: str, horizon: int = 24) -> np.ndarray:
        last_val = float(history_df[target_col].iloc[-1])
        return np.full(horizon, last_val, dtype=float)


class MovingAverageForecaster(BaseForecaster):
    """Forecasts the rolling mean of the last `window` observations across all horizon steps."""

    def __init__(self, window: int = 6):
        self.window = window

    def fit(self, train_df: pd.DataFrame, target_col: str) -> MovingAverageForecaster:
        self.target_col = target_col
        return self

    def predict_horizon(self, history_df: pd.DataFrame, target_col: str, horizon: int = 24) -> np.ndarray:
        window_vals = history_df[target_col].iloc[-self.window :]
        pred_val = float(window_vals.mean())
        return np.full(horizon, pred_val, dtype=float)


class HistoricalMeanForecaster(BaseForecaster):
    """Forecasts the historical mean across the entire training period."""

    def __init__(self):
        self.mean_val = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str) -> HistoricalMeanForecaster:
        self.mean_val = float(train_df[target_col].mean())
        return self

    def predict_horizon(self, history_df: pd.DataFrame, target_col: str, horizon: int = 24) -> np.ndarray:
        return np.full(horizon, self.mean_val, dtype=float)


class MeanRevertingForecaster(BaseForecaster):
    """Mean-reverting autoregressive forecaster: y_{t+h} = mu + phi^h * (y_t - mu).

    Models short-term persistence decaying towards the unconditional historical mean
    as the forecast horizon h expands and temporal autocorrelation decays to zero.
    """

    def __init__(self, phi: float = 0.90):
        self.phi = phi
        self.mu = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str) -> MeanRevertingForecaster:
        self.target_col = target_col
        self.mu = float(train_df[target_col].mean())
        return self

    def predict_horizon(self, history_df: pd.DataFrame, target_col: str, horizon: int = 24) -> np.ndarray:
        y_last = float(history_df[target_col].iloc[-1])
        steps = np.arange(1, horizon + 1)
        preds = self.mu + (self.phi ** steps) * (y_last - self.mu)
        return preds.astype(float)


class DirectMultiHorizonForecaster(BaseForecaster):
    """Direct multi-step forecaster.

    Fits an independent estimator for each horizon step h in 1..horizon.
    Avoids error accumulation inherent in recursive autoregressive forecasting.
    """

    def __init__(
        self,
        model_type: Literal["ridge", "gbm"] = "ridge",
        alpha: float = 1.0,
        horizon: int = 24,
        min_history: int = 12,
        quantile: float | None = None,
    ):
        self.model_type = model_type
        self.alpha = alpha
        self.horizon = horizon
        self.min_history = min_history
        self.quantile = quantile
        self.models: dict[int, Pipeline | HistGradientBoostingRegressor] = {}
        self.feature_names: list[str] = []

    def fit(self, train_df: pd.DataFrame, target_col: str) -> DirectMultiHorizonForecaster:
        self.target_col = target_col
        X, Y = build_direct_forecasting_matrices(
            train_df, target_col=target_col, horizon=self.horizon, min_history=self.min_history
        )
        self.feature_names = list(X.columns)

        for h in range(1, self.horizon + 1):
            y_h = Y[f"h_{h}"].values
            if self.model_type == "ridge":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=self.alpha, random_state=42)),
                ])
                model.fit(X, y_h)
            elif self.model_type == "gbm":
                if self.quantile is not None:
                    model = HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=self.quantile,
                        max_iter=50,
                        max_depth=3,
                        min_samples_leaf=5,
                        l2_regularization=self.alpha,
                        random_state=42,
                    )
                else:
                    model = HistGradientBoostingRegressor(
                        loss="squared_error",
                        max_iter=50,
                        max_depth=3,
                        min_samples_leaf=5,
                        l2_regularization=self.alpha,
                        random_state=42,
                    )
                model.fit(X, y_h)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            self.models[h] = model

        return self

    def predict_horizon(self, history_df: pd.DataFrame, target_col: str, horizon: int = 24) -> np.ndarray:
        feat_dict = extract_origin_features(history_df, target_col=target_col)
        x_origin = pd.DataFrame([feat_dict])[self.feature_names]

        preds = np.zeros(horizon, dtype=float)
        for h in range(1, horizon + 1):
            if h in self.models:
                preds[h - 1] = float(self.models[h].predict(x_origin)[0])
            else:
                preds[h - 1] = float(self.models[max(self.models.keys())].predict(x_origin)[0])

        return preds


class WorkloadForecaster:
    """Unified pipeline producing compliant forecasts for both:
    1. active_vm_count
    2. aggregate_cpu_utilization
    """

    def __init__(
        self,
        cpu_model_type: Literal["mean_reverting", "ridge", "gbm", "persistence", "moving_average"] = "mean_reverting",
        alpha: float = 1.0,
        horizon: int = 24,
        phi: float = 0.90,
    ):
        self.horizon = horizon
        self.cpu_model_type = cpu_model_type

        if cpu_model_type == "persistence":
            self.cpu_model = PersistenceForecaster()
        elif cpu_model_type == "moving_average":
            self.cpu_model = MovingAverageForecaster(window=6)
        elif cpu_model_type == "mean_reverting":
            self.cpu_model = MeanRevertingForecaster(phi=phi)
        else:
            self.cpu_model = DirectMultiHorizonForecaster(
                model_type=cpu_model_type, alpha=alpha, horizon=horizon
            )

        # For active_vm_count: persistence/known capacity forecaster
        self.vm_model = PersistenceForecaster()

    def fit(self, train_df: pd.DataFrame) -> WorkloadForecaster:
        self.train_df = train_df.copy()
        self.cpu_model.fit(train_df, "aggregate_cpu_utilization")
        self.vm_model.fit(train_df, "active_vm_count")
        return self

    def forecast(self, history_df: pd.DataFrame, holdout_timestamps: list[int | float]) -> pd.DataFrame:
        """Produce the submission DataFrame matching SUBMISSION.md schema."""
        horizon = len(holdout_timestamps)
        cpu_preds = self.cpu_model.predict_horizon(history_df, "aggregate_cpu_utilization", horizon=horizon)
        vm_preds = self.vm_model.predict_horizon(history_df, "active_vm_count", horizon=horizon)

        # Enforce physical constraints:
        # active_vm_count >= 0
        vm_preds = np.maximum(vm_preds, 0.0)
        # aggregate_cpu_utilization in [0, 100]
        cpu_preds = np.clip(cpu_preds, 0.0, 100.0)

        df_out = pd.DataFrame({
            "timestamp": holdout_timestamps,
            "active_vm_count": vm_preds,
            "aggregate_cpu_utilization": cpu_preds,
        })
        return df_out
