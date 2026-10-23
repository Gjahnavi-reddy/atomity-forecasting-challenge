"""Benchmark script for evaluating forecasting models.

Runs:
1. Leakage-free expanding-window CV on the 110 training observations (5 folds of horizon 24).
2. Final evaluation against the 24-step holdout horizon.
3. Generates comparison tables and publication plots for the report.
"""
from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from cwfc.data import load_raw_data, build_aggregate_targets, get_train_holdout_split
from cwfc.models import (
    PersistenceForecaster,
    MovingAverageForecaster,
    HistoricalMeanForecaster,
    MeanRevertingForecaster,
    DirectMultiHorizonForecaster,
    WorkloadForecaster,
)
from cwfc.validation import evaluate_model_rolling_cv
from cwfc.metrics import mae, rmse, smape, normalized_mae, max_error, peak_mae


def run_benchmarks(data_dir: str = "challenge_data", output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    print("Loading dataset...")
    df_cpu, df_meta = load_raw_data(data_dir=data_dir)
    df_agg = build_aggregate_targets(df_cpu, df_meta=df_meta)

    train_df, holdout_df = get_train_holdout_split(df_agg, train_size=110, holdout_size=24)
    target_col = "aggregate_cpu_utilization"

    print(f"Train records: {len(train_df)}, Holdout records: {len(holdout_df)}")

    # Models to benchmark
    models = {
        "Persistence": PersistenceForecaster(),
        "Moving_Average_30m": MovingAverageForecaster(window=6),
        "Historical_Mean": HistoricalMeanForecaster(),
        "Mean_Reverting_AR": MeanRevertingForecaster(phi=0.90),
        "Direct_Ridge": DirectMultiHorizonForecaster(model_type="ridge", alpha=10.0, horizon=24, min_history=12),
        "Direct_GBDT": DirectMultiHorizonForecaster(model_type="gbm", alpha=1.0, horizon=24, min_history=12),
    }

    base_pers = PersistenceForecaster()

    # 1. Rolling-origin cross-validation on the 110 train points
    print("\n" + "=" * 60)
    print("RUNNING 5-FOLD EXPANDING-WINDOW TEMPORAL CV (Horizon = 24)")
    print("=" * 60)

    cv_results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        res = evaluate_model_rolling_cv(
            forecaster=model,
            train_df=train_df,
            target_col=target_col,
            horizon=24,
            min_train_size=42,
            step=11,
            baseline_forecaster=base_pers,
        )
        cv_results[name] = res
        print(
            f"  -> MAE: {res['mean_mae']:.4f} ± {res['std_mae']:.4f} | "
            f"RMSE: {res['mean_rmse']:.4f} ± {res['std_rmse']:.4f} | "
            f"Norm MAE: {res.get('mean_normalized_mae', 1.0):.4f}"
        )

    # 2. Final Fit on Train (110 obs) and Evaluation on Holdout (24 obs)
    print("\n" + "=" * 60)
    print("EVALUATING ON THE 24-STEP HOLDOUT HORIZON (t=111..134)")
    print("=" * 60)

    y_holdout_true = holdout_df[target_col].values
    holdout_timestamps = holdout_df["timestamp"].values
    holdout_hours = holdout_timestamps / 3600.0

    holdout_preds = {}
    holdout_metrics = {}

    # Get persistence prediction for normalized MAE
    base_pers.fit(train_df, target_col)
    y_base_holdout = base_pers.predict_horizon(train_df, target_col, horizon=24)

    for name, model in models.items():
        model.fit(train_df, target_col)
        y_pred = model.predict_horizon(train_df, target_col, horizon=24)
        holdout_preds[name] = y_pred

        m_mae = mae(y_holdout_true, y_pred)
        m_rmse = rmse(y_holdout_true, y_pred)
        m_smape = smape(y_holdout_true, y_pred)
        m_max_err = max_error(y_holdout_true, y_pred)
        m_peak_mae = peak_mae(y_holdout_true, y_pred, percentile=90.0)
        m_norm_mae = normalized_mae(y_holdout_true, y_pred, y_base_holdout)

        holdout_metrics[name] = {
            "mae": float(m_mae),
            "rmse": float(m_rmse),
            "smape": float(m_smape),
            "max_error": float(m_max_err),
            "peak_mae": float(m_peak_mae),
            "normalized_mae": float(m_norm_mae),
        }
        print(
            f"{name:20s} | MAE: {m_mae:.4f} | RMSE: {m_rmse:.4f} | "
            f"SMAPE: {m_smape:.2f}% | Norm MAE: {m_norm_mae:.4f} | Peak MAE: {m_peak_mae:.4f}"
        )

    # Train Quantile Forecaster (P90) for operational capacity planning
    print("\nTraining Quantile Model (P90) for operational safety headroom...")
    p90_model = DirectMultiHorizonForecaster(
        model_type="gbm", alpha=1.0, horizon=24, min_history=15, quantile=0.90
    )
    p90_model.fit(train_df, target_col)
    y_p90_pred = p90_model.predict_horizon(train_df, target_col, horizon=24)
    holdout_preds["GBDT_P90_Upper"] = y_p90_pred

    # -------------------------------------------------------------
    # Figure 4: Horizon Step Degradation (CV)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    h_steps = np.arange(1, 25)
    for name in ["Persistence", "Moving_Average_30m", "Historical_Mean", "Mean_Reverting_AR", "Direct_Ridge", "Direct_GBDT"]:
        ax.plot(h_steps, cv_results[name]["step_mae_by_horizon"], marker="o", ms=4, lw=1.8, label=name.replace("_", " "))

    ax.set_title("Forecast Error (MAE) by Horizon Step Across Temporal CV Folds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Horizon Step h (5-minute intervals, 1 to 24 = 2 Hours)")
    ax.set_ylabel("Cross-Validation MAE (% CPU)")
    ax.set_xticks(range(2, 25, 2))
    ax.legend(framealpha=0.9, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig4_path = os.path.join(output_dir, "model_horizon_comparison.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved {fig4_path}")

    # -------------------------------------------------------------
    # Figure 5: Holdout Forecast Trajectory vs Ground Truth
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    # Context: last 12 points of train window
    context_tail = train_df.iloc[-12:]
    ax.plot(context_tail["timestamp"] / 3600.0, context_tail[target_col], "k-o", ms=4, lw=1.5, label="Observed History (t=99..110)")
    
    # Ground truth holdout
    ax.plot(holdout_hours, y_holdout_true, "g-s", ms=5, lw=2.2, label="Actual Ground Truth Holdout")
    
    # Model predictions
    ax.plot(holdout_hours, holdout_preds["Persistence"], "r--", lw=1.8, label="Persistence (Last Value)")
    ax.plot(holdout_hours, holdout_preds["Moving_Average_30m"], color="purple", ls="-.", lw=1.8, label="Moving Average (30m)")
    ax.plot(holdout_hours, holdout_preds["Mean_Reverting_AR"], color="#1f77b4", lw=2.4, label="Mean-Reverting AR (Primary)")
    ax.plot(holdout_hours, holdout_preds["Direct_Ridge"], color="#17becf", ls="--", lw=1.8, label="Direct Ridge")
    ax.plot(holdout_hours, holdout_preds["Direct_GBDT"], color="#ff7f0e", ls=":", lw=2.0, label="Direct GBDT")
    
    # Upper P90 headroom
    ax.fill_between(holdout_hours, holdout_preds["Mean_Reverting_AR"], y_p90_pred, color="#1f77b4", alpha=0.18, label="Operational P90 Headroom Buffer")
    ax.plot(holdout_hours, y_p90_pred, color="#1f77b4", ls=":", lw=1.5, label="P90 Upper Capacity Bound")

    ax.axvline(110 * 300 / 3600.0, color="gray", ls=":", lw=1.5)
    ax.set_title("Holdout Evaluation: Forecast Trajectories vs Actual Ground Truth Demand", fontsize=12, fontweight="bold")
    ax.set_xlabel("Elapsed Time (Hours)")
    ax.set_ylabel("Aggregate CPU Utilization (%)")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig5_path = os.path.join(output_dir, "holdout_forecast_comparison.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved {fig5_path}")

    # Save comprehensive results JSON
    benchmark_payload = {
        "cross_validation": {k: {m: v for m, v in val.items() if m != "step_mae_by_horizon"} for k, val in cv_results.items()},
        "holdout": holdout_metrics,
    }
    bench_json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(bench_json_path, "w") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"Saved benchmark numbers to {bench_json_path}")

    # Also generate the official predictions.csv using the Primary WorkloadForecaster (Mean Reverting)
    print("\nGenerating final official predictions.csv...")
    wf = WorkloadForecaster(cpu_model_type="mean_reverting", phi=0.90, horizon=24)
    wf.fit(train_df)
    pred_df = wf.forecast(train_df, holdout_timestamps.tolist())
    pred_df.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv successfully. First 5 rows:")
    print(pred_df.head(5))

    return benchmark_payload


if __name__ == "__main__":
    run_benchmarks()
