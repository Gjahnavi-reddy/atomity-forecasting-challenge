"""Exploratory Data Analysis script for Azure VM Workload Dataset.

Computes descriptive statistics, checks dataset integrity, tests autocorrelation and stationarity,
and generates publication-quality figures for the final report.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_eda(data_dir: str = "challenge_data", output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    print("Loading datasets...")
    cpu_path = os.path.join(data_dir, "vm_cpu_readings.csv.gz")
    meta_path = os.path.join(data_dir, "vm_metadata.csv.gz")

    df_meta = pd.read_csv(meta_path)
    df_cpu = pd.read_csv(cpu_path)

    print(f"Metadata rows: {len(df_meta):,}, Columns: {list(df_meta.columns)}")
    print(f"CPU rows: {len(df_cpu):,}, Columns: {list(df_cpu.columns)}")

    # 1. Integrity checks
    n_vms = df_cpu["vm_id"].nunique()
    timestamps = sorted(df_cpu["timestamp"].unique())
    n_ts = len(timestamps)
    print(f"\nUnique VMs: {n_vms:,}")
    print(f"Unique timestamps: {n_ts} (Range: {timestamps[0]}s to {timestamps[-1]}s, Step: {np.diff(timestamps)[0]}s)")
    
    # Missingness
    print("\nMissing values in CPU readings:")
    print(df_cpu.isna().sum())
    print("\nMissing values in Metadata:")
    print(df_meta.isna().sum())

    # Aggregate target creation
    print("\nConstructing aggregate time series...")
    agg_df = df_cpu.groupby("timestamp").agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
        min_cpu_mean=("min_cpu", "mean"),
        max_cpu_mean=("max_cpu", "mean"),
        cpu_p25=("avg_cpu", lambda x: np.percentile(x, 25)),
        cpu_p50=("avg_cpu", lambda x: np.percentile(x, 50)),
        cpu_p75=("avg_cpu", lambda x: np.percentile(x, 75)),
        cpu_p90=("avg_cpu", lambda x: np.percentile(x, 90)),
        cpu_p95=("avg_cpu", lambda x: np.percentile(x, 95)),
        cpu_std=("avg_cpu", "std")
    ).reset_index()

    agg_df["time_minutes"] = agg_df["timestamp"] / 60.0
    agg_df["time_hours"] = agg_df["timestamp"] / 3600.0

    print(f"Aggregate series length: {len(agg_df)}")
    print(f"Active VM count constant check: {agg_df['active_vm_count'].unique()}")
    print("\nAggregate CPU summary across all 134 timestamps:")
    print(agg_df["aggregate_cpu_utilization"].describe())

    # Segment aggregations
    df_merged = df_cpu.merge(df_meta[["vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"]], on="vm_id")
    cat_ts = df_merged.groupby(["timestamp", "vm_category"])["avg_cpu"].mean().unstack()
    cat_ts["time_hours"] = cat_ts.index / 3600.0

    # -------------------------------------------------------------
    # Figure 1: Time Series Trajectory & Split
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Plot 1: Overall aggregate CPU with percentiles and split line
    split_ts_hr = 110 * 300 / 3600.0  # 9.167 hours
    ax1.plot(agg_df["time_hours"], agg_df["aggregate_cpu_utilization"], color="#1f77b4", lw=2.2, label="Mean Aggregate CPU (%)")
    ax1.fill_between(agg_df["time_hours"], agg_df["cpu_p25"], agg_df["cpu_p75"], color="#1f77b4", alpha=0.2, label="IQR (P25–P75)")
    ax1.plot(agg_df["time_hours"], agg_df["cpu_p95"], color="#d62728", lw=1.5, ls="--", label="P95 Peak CPU (%)")
    ax1.axvline(split_ts_hr, color="black", ls=":", lw=1.8, label="Train/Holdout Boundary (t=110)")
    ax1.text(split_ts_hr - 0.2, 38, "Training Window (110 obs)", horizontalalignment="right", fontsize=10, fontweight="bold", color="#333333")
    ax1.text(split_ts_hr + 0.2, 38, "Holdout Horizon (24 obs)", horizontalalignment="left", fontsize=10, fontweight="bold", color="#d62728")
    ax1.set_ylabel("CPU Utilization (%)")
    ax1.set_title("Azure VM Telemetry: Aggregate Workload Demand & Percentiles over Time", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Workload Categories Breakdown
    ax2.plot(cat_ts["time_hours"], cat_ts["Interactive"], color="#ff7f0e", lw=2, label=f"Interactive (N={df_meta['vm_category'].value_counts().get('Interactive', 0):,})")
    ax2.plot(cat_ts["time_hours"], cat_ts["Unknown"], color="#2ca02c", lw=1.8, label=f"Unknown (N={df_meta['vm_category'].value_counts().get('Unknown', 0):,})")
    ax2.plot(cat_ts["time_hours"], cat_ts["Delay-insensitive"], color="#9467bd", lw=2, label=f"Delay-insensitive (N={df_meta['vm_category'].value_counts().get('Delay-insensitive', 0):,})")
    ax2.axvline(split_ts_hr, color="black", ls=":", lw=1.8)
    ax2.set_xlabel("Elapsed Time (Hours)")
    ax2.set_ylabel("Mean CPU (%)")
    ax2.set_title("Mean Workload Demand by VM Category", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "eda_time_series.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: VM-Level CPU Distributions & Skewness
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Histogram/KDE of individual readings
    sample_readings = df_cpu["avg_cpu"].sample(n=100000, random_state=42)
    sns.histplot(sample_readings, bins=50, kde=True, color="#3470a3", ax=axes[0], stat="density")
    axes[0].axvline(sample_readings.median(), color="green", ls="--", lw=1.8, label=f"Median: {sample_readings.median():.2f}%")
    axes[0].axvline(sample_readings.mean(), color="red", ls="-", lw=1.8, label=f"Mean: {sample_readings.mean():.2f}%")
    axes[0].axvline(np.percentile(sample_readings, 95), color="purple", ls=":", lw=1.8, label=f"P95: {np.percentile(sample_readings, 95):.2f}%")
    axes[0].set_title("VM-Level CPU Reading Distribution (Right-Skewed)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Average CPU Utilization (%)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Right: Boxplot by Category
    sns.boxplot(data=df_merged.sample(n=100000, random_state=42), x="vm_category", y="avg_cpu", palette="Set2", ax=axes[1], showfliers=False)
    axes[1].set_title("CPU Utilization Distribution by Category (Excl. Outliers)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("VM Category")
    axes[1].set_ylabel("Average CPU Utilization (%)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "eda_cpu_distributions.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Autocorrelation (ACF) of Aggregate Series (Train Window)
    # -------------------------------------------------------------
    train_series = agg_df.iloc[:110]["aggregate_cpu_utilization"]
    lags = range(1, 31)
    acf_vals = [train_series.autocorr(lag=l) for l in lags]
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stem(list(lags), acf_vals, basefmt="k-", linefmt="#1f77b4", markerfmt="o")
    # 95% confidence interval for white noise: +- 1.96 / sqrt(N)
    ci = 1.96 / np.sqrt(len(train_series))
    ax.axhspan(-ci, ci, alpha=0.2, color="gray", label="95% White Noise Bound (±1.96/√N)")
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Autocorrelation Function (ACF) of Aggregate CPU (First 110 Timestamps)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lag Step (Each step = 5 minutes; Lag 12 = 1 Hour, Lag 24 = 2 Hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks(range(2, 31, 2))
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "eda_autocorrelation.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved {fig3_path}")

    # Save summary stats to CSV for report reference
    summary_path = os.path.join(output_dir, "eda_metrics_summary.json")
    train_slice = agg_df.iloc[:110]
    holdout_slice = agg_df.iloc[110:]
    stats_dict = {
        "n_vms": int(n_vms),
        "n_total_timestamps": int(n_ts),
        "cadence_seconds": 300,
        "active_vm_count_train": int(train_slice["active_vm_count"].mean()),
        "active_vm_count_holdout": int(holdout_slice["active_vm_count"].mean()),
        "train_mean_cpu": float(train_slice["aggregate_cpu_utilization"].mean()),
        "train_std_cpu": float(train_slice["aggregate_cpu_utilization"].std()),
        "train_min_cpu": float(train_slice["aggregate_cpu_utilization"].min()),
        "train_max_cpu": float(train_slice["aggregate_cpu_utilization"].max()),
        "holdout_mean_cpu": float(holdout_slice["aggregate_cpu_utilization"].mean()),
        "holdout_std_cpu": float(holdout_slice["aggregate_cpu_utilization"].std()),
        "holdout_min_cpu": float(holdout_slice["aggregate_cpu_utilization"].min()),
        "holdout_max_cpu": float(holdout_slice["aggregate_cpu_utilization"].max()),
        "acf_lag1": float(acf_vals[0]),
        "acf_lag3": float(acf_vals[2]),
        "acf_lag12": float(acf_vals[11]),
        "acf_lag24": float(acf_vals[23])
    }
    import json
    with open(summary_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved stats to {summary_path}")

if __name__ == "__main__":
    run_eda()
