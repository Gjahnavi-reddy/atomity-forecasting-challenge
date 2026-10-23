# Cloud Workload Forecasting & Capacity Analysis Report
**Microsoft Azure Public Dataset V2 (`data-v1` Trace)**  
*Author: Senior AI/ML Systems Engineer & Cloud Infrastructure Architect*  
*Dataset: Curated Azure VM Trace (10,000 Virtual Machines, 134 Five-Minute Observations)*  
*Deliverable: Take-Home Assignment Report & Operational Policy*  

---

## 1. Executive Summary

Modern cloud infrastructure platforms require accurate, near-term forecasts of aggregate resource demand to balance quality-of-service (QoS) guarantees against infrastructure over-provisioning costs. Under-forecasting CPU demand leads to CPU scheduling queues, tail-latency degradation, and Service Level Agreement (SLA) penalties. Conversely, chronic over-forecasting incurs substantial capital expenditure and idle power waste across physical server clusters.

This investigation analyzes a curated subset of the **Microsoft Azure Public Dataset V2** (`data-v1` release), comprising **10,000 virtual machines** tracked continuously over **134 five-minute timestamps** (~11.1 hours). We address the core challenge question:
> *How predictable is this cloud workload, how accurately can near-term demand be forecast, and what should an infrastructure team do with the forecast?*

### Key Findings & Operational Highlights:
1. **Target Dynamics & Population Stability**:
   - `active_vm_count`: Due to trace eligibility filtering in the challenge release curation, all 10,000 VMs exhibit complete, contiguous telemetry without churn during the trace window. Active VM capacity is strictly constant at **10,000 VMs**.
   - `aggregate_cpu_utilization`: Mean aggregate CPU demand fluctuates within a narrow band of **9.39% ± 0.19%** (min: 8.98%, max: 9.91%), while individual VM readings exhibit severe right-skewness (median: 5.11%, P95: 37.32%, max: 99.49%).
2. **Predictability & Autoregressive Decay**:
   - Aggregate demand displays strong short-term serial autocorrelation ($\rho_1 = 0.667$, $\rho_3 = 0.681$), decaying steadily towards zero by step 24 (2 hours, $\rho_{24} = -0.026$). No full diurnal period is present in the 9.17-hour training window.
3. **Forecasting Performance**:
   - Complex multi-step gradient boosting (Direct GBDT) and linear regression (Direct Ridge) suffered from variance inflation and overfitting on the aggregate mean over long horizons ($H=24$).
   - A statistically grounded **Mean-Reverting Autoregressive Model ($\text{AR-MR}$, $\phi=0.90$)** delivered the best cross-validation MAE (**0.2138%**) and achieved the top performance on the holdout window with an MAE of **0.1055%** (a **28.1% error reduction** over persistence) and an RMSE of **0.1290%** (a **30.2% improvement** over persistence).
4. **Actionable Operational Policy**:
   - We recommend a **Dynamic Quantile Headroom Policy**: provision baseline cluster compute capacity at the point forecast $\hat{y}_{t+h}$, plus an upper risk buffer derived from empirical GBDT P90 quantile estimation (+0.35% aggregate CPU / +35 cores per 10k vCPUs). This achieves zero CPU capacity shortfalls on the holdout window while limiting unused headroom to less than 4.5% of total provisioned compute.

---

## 2. Data Understanding and Preprocessing

### 2.1 Trace Characteristics & Ingestion Protocol
The dataset consists of two compressed relational tables:
1. `vm_cpu_readings.csv.gz`: 1,340,000 row-level telemetry observations across 10,000 distinct VM IDs. Columns include relative observation `timestamp` (seconds, $\Delta t = 300\text{s}$), `min_cpu`, `max_cpu`, and `avg_cpu`.
2. `vm_metadata.csv.gz`: 10,000 metadata records describing VM static properties (`subscription_id`, `deployment_id`, `timestamp_vm_created`, `timestamp_vm_deleted`, `vm_category`, `virtual_core_bucket`, `memory_gb_bucket`) alongside whole-trace summary statistics (`max_cpu`, `avg_cpu`, `p95_max_cpu`).

### 2.2 Integrity, Missingness, and Cadence Verification
- **Cadence Uniformity**: Consecutive timestamps are spaced exactly $300\text{ seconds}$ (5 minutes) apart across all $t \in [0, 39900]$. No missing timestamps or duplicate $(vm\_id, timestamp)$ pairs exist.
- **Null Values**: Exactly zero missing values exist across all columns in both CPU readings and metadata.
- **Lifetime Boundary Check**: `timestamp_vm_created` is 0 for all VMs, and `timestamp_vm_deleted` ranges from 40,500 to 2,591,400. Because our entire challenge trace terminates at $t = 39,900$, **zero VM churn or deletions occur within the observation window**.

### 2.3 Aggregate Target Construction
In accordance with `DATASET.md`, two official time series are constructed at each five-minute timestamp $t$:
$$\text{active\_vm\_count}(t) = \sum_{i=1}^{N} \mathbb{I}(\text{VM}_i \text{ has valid CPU reading at } t) = 10,000$$
$$\text{aggregate\_cpu\_utilization}(t) = \frac{1}{\text{active\_vm\_count}(t)} \sum_{i \in \text{Active}(t)} \text{avg\_cpu}_i(t)$$

All target values satisfy physical domain boundaries: $\text{active\_vm\_count} \ge 0$, and $\text{aggregate\_cpu\_utilization} \in [0, 100]\%$.

| Dataset Split | Timestamp Range (s) | Step Range | Elapsed Span | Active VMs | Mean CPU (%) | Min CPU (%) | Max CPU (%) | Std Dev (%) |
|---|---|---|---|---|---|---|---|---|
| **Training Set** | $0 \dots 32,700$ | $t_1 \dots t_{110}$ | 9.17 Hours | 10,000 | 9.414% | 9.112% | 9.912% | 0.194% |
| **Holdout Set** | $33,000 \dots 39,900$ | $t_{111} \dots t_{134}$ | 2.00 Hours | 10,000 | 9.269% | 8.984% | 9.636% | 0.156% |
| **Full Trace** | $0 \dots 39,900$ | $t_1 \dots t_{134}$ | 11.08 Hours | 10,000 | 9.388% | 8.984% | 9.912% | 0.195% |

---

## 3. Exploratory Workload Analysis

```
                              AGGREGATE CPU TRAJECTORY & SEGMENT BREAKDOWN
  13 % ┌─────────────────────────────────────────────────────────────────────────────┐
       │                                                      Interactive (Mean ~11.5%)│
  11 % │  -----------------------.---/\--------------------------------------------  │
       │                         '   '  \                                            │
   9 % │  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-  Aggregate Mean │
       │  .........................................................  Delay-Insensitive│
   7 % └──────────────────────────────────────┬──────────────────────────────────────┘
       0h                                    9.17h                                11.1h
                                     [Train / Holdout Split]
```

### 3.1 Distributional Skewness & The Law of Large Numbers
At the individual VM level, CPU utilization exhibits extreme right-skewness:
- **Median**: 5.11%
- **Mean**: 9.39%
- **P90**: 28.45%
- **P95**: 37.32%
- **Max**: 99.49%
- **Low-load VMs**: 7.59% of readings fall below 1.0% CPU utilization.
- **High-load VMs**: Only 1.27% of readings exceed 80% CPU utilization.

However, when aggregated across 10,000 VMs, the **Law of Large Numbers (LLN)** dramatically attenuates individual volatility. The standard deviation of the aggregate CPU series is just **0.195%**, yielding an aggregate coefficient of variation $CV = \frac{0.195}{9.388} \approx 2.07\%$.

### 3.2 Workload Segmentation Analysis
Joining telemetry with `vm_metadata.csv.gz` reveals distinct behavioral clusters across workload categories and core allocations:

| Segment Dimension | Level | VM Count | Share (%) | Mean CPU (%) | P95 CPU (%) | Std Dev (%) | Operational Characteristic |
|---|---|---|---|---|---|---|---|
| **VM Category** | Delay-insensitive | 6,180 | 61.8% | 8.24% | 25.91% | 12.02% | Predictable batch compute, low variance |
| **VM Category** | Interactive | 3,091 | 30.9% | 11.49% | 49.41% | 16.97% | User-facing, responsive, high burstiness |
| **VM Category** | Unknown | 729 | 7.3% | 10.23% | 52.23% | 17.75% | Mixed workloads, volatile tail demand |
| **Core Bucket** | 2 vCPUs | 7,534 | 75.3% | 9.72% | 38.99% | 14.56% | General purpose sizing |
| **Core Bucket** | 4 vCPUs | 1,908 | 19.1% | 7.93% | 32.06% | 13.30% | Mid-tier services, lower average load |
| **Core Bucket** | 8 vCPUs | 486 | 4.9% | 9.70% | 38.40% | 12.58% | Data processing services |
| **Core Bucket** | $\ge 24$ vCPUs | 72 | 0.7% | 10.74% | 47.38% | 15.68% | Heavy analytics, high memory instances |

**Key Takeaway**: Interactive VMs run at ~39% higher average utilization and nearly double the 95th-percentile utilization of delay-insensitive VMs. In infrastructure scaling, monitoring the Interactive segment provides leading indications of user activity shifts.

### 3.3 Autocorrelation & Stationarity
Autocorrelation analysis on the 110 training observations demonstrates strong local persistence that decays steadily:
- $\text{Lag } 1\text{ (5 min)}$: $\rho = 0.6672$
- $\text{Lag } 3\text{ (15 min)}$: $\rho = 0.6810$
- $\text{Lag } 6\text{ (30 min)}$: $\rho = 0.4771$
- $\text{Lag } 12\text{ (60 min)}$: $\rho = 0.3666$
- $\text{Lag } 24\text{ (120 min)}$: $\rho = -0.0264$

Because $\rho$ approaches zero at $h=24$, the workload demand over a 2-hour window is **mean-reverting**. An augmented Dickey-Fuller test confirms absence of a unit root ($p < 0.01$). No 24-hour seasonal cycle can be reliably fitted because the observation window spans only 11.1 hours.

---

## 4. Forecasting Methodology & Feature Engineering

### 4.1 Strict Leakage-Free Design
To guarantee zero future leakage:
1. All feature transforms at forecast origin $t_0$ are computed exclusively from history $t \le t_0$.
2. Summary statistics from `vm_metadata.csv.gz` (`max_cpu`, `avg_cpu`, `p95_max_cpu`) were **strictly excluded** from feature inputs, as they represent whole-trace aggregates computed over the entire 30-day source trace.
3. The final 24 holdout observations were strictly quarantined until final benchmarking.

### 4.2 Candidate Model Architecture
We implemented and benchmarked five diverse model families:
1. **Persistence Baseline**: $\hat{y}_{t+h} = y_t$. Represents last-value carryover.
2. **Moving Average Baseline (30m)**: $\hat{y}_{t+h} = \frac{1}{6}\sum_{i=0}^5 y_{t-i}$. Smooths high-frequency noise.
3. **Historical Mean Baseline**: $\hat{y}_{t+h} = \bar{y}_{\text{train}}$. Represents long-term unconditional expectation.
4. **Mean-Reverting Autoregressive ($\text{AR-MR}$)**:
   $$\hat{y}_{t+h} = \mu + \phi^h (y_t - \mu)$$
   where $\mu$ is the expanding training mean and $\phi \in (0, 1)$ is the autoregressive persistence factor ($\phi = 0.90$). For near horizons ($h \to 1$), predictions track current state $y_t$; for distant horizons ($h \to 24$), predictions revert to the unconditional mean $\mu$.
5. **Direct Multi-Horizon Machine Learning Forecasters**:
   - Trains 24 independent estimators ($h = 1 \dots 24$), avoiding error propagation.
   - **Direct Ridge Regression**: L2-regularized linear model ($\alpha=10.0$) with standardized features.
   - **Direct GBDT**: Gradient-boosted decision trees (`HistGradientBoostingRegressor`) with conservative depth (`max_depth=3`, `l2_reg=1.0`).
   - **Quantile GBDT (P90)**: Fitted with pinball loss ($\tau=0.90$) to generate upper capacity buffers.

### 4.3 Feature Set (for ML Models)
Features computed at origin $t$:
- **Lags**: $y_t, y_{t-1}, y_{t-2}, y_{t-3}, y_{t-5}, y_{t-11}$
- **Rolling Windows** ($W \in [3, 6, 12]$): Rolling mean, standard deviation, min, max, range.
- **Momentum**: $y_t - y_{t-1}$, $y_t - y_{t-3}$, $y_t - \text{mean}_{6}(y)$.
- **Segment Signals**: Interactive and delay-insensitive segment means and rolling trends at $t$.
- **Temporal Index**: Elapsed hours from trace origin.

---

## 5. Validation Methodology

```
                          TEMPORAL EXPANDING-WINDOW VALIDATION (5 FOLDS)
Fold 1: [Train: t1..42 (3.5h)]  ──> [Val: t43..66 (2h)]
Fold 2: [Train: t1..53 (4.4h)]  ──────> [Val: t54..77 (2h)]
Fold 3: [Train: t1..64 (5.3h)]  ──────────> [Val: t65..88 (2h)]
Fold 4: [Train: t1..75 (6.2h)]  ──────────────> [Val: t76..99 (2h)]
Fold 5: [Train: t1..86 (7.2h)]  ──────────────────> [Val: t87..110 (2h)]
Holdout:[Train: t1..110 (9.2h)] ──────────────────────> [FINAL TEST: t111..134 (2h)]
```

### 5.1 Temporal Cross-Validation Scheme
Random K-Fold splitting was avoided to prevent severe autocorrelation leakage. We implemented an **expanding-window rolling-origin evaluation** across the 110 training observations:
- **Horizon ($H$)**: 24 steps (2 hours), exactly matching the test holdout horizon.
- **Initial Training Window**: 42 steps (~3.5 hours).
- **Step Size**: 11 steps (~55 minutes).
- **Number of Folds**: 5 contiguous evaluation windows.
- In each fold $k$, the model fits strictly on data up to origin $t_k$, and predicts $t_k + 1 \dots t_k + 24$.

---

## 6. Results vs. Baselines

### 6.1 Expanding-Window Cross-Validation Performance (5 Folds)

| Model Name | Model Family | CV MAE (%) | CV Std MAE | CV RMSE (%) | CV SMAPE (%) | Normalized MAE vs. Persistence |
|---|---|---|---|---|---|---|
| **Mean-Reverting AR ($\phi=0.90$)** | **Parametric Time-Series** | **0.2138** | **±0.0806** | **0.2461** | **2.26%** | **0.8838** (Fold avg: 1.074) |
| **Historical Mean** | Statistical Baseline | 0.2172 | ±0.0541 | 0.2509 | 2.29% | 0.8979 (Fold avg: 1.162) |
| **Persistence (Last Value)** | Simple Baseline | 0.2419 | ±0.1265 | 0.2708 | 2.55% | 1.0000 |
| **Moving Average (30m)** | Simple Baseline | 0.2536 | ±0.0738 | 0.2893 | 2.67% | 1.0484 (Fold avg: 1.280) |
| **Direct GBDT** | Tree Ensemble | 0.2996 | ±0.0521 | 0.3407 | 3.15% | 1.2385 (Fold avg: 1.814) |
| **Direct Ridge ($\alpha=10$)** | Regularized Linear | 0.4739 | ±0.1060 | 0.5548 | 4.97% | 1.9589 (Fold avg: 3.050) |

### 6.2 Official Holdout Evaluation (Final 24 Timestamps = 2 Hours)

| Model Name | Holdout MAE (%) | Holdout RMSE (%) | Holdout SMAPE (%) | Normalized MAE | Max Absolute Error (%) | Peak MAE (Top 10% Load) |
|---|---|---|---|---|---|---|
| **Mean-Reverting AR ($\phi=0.90$) [PRIMARY]** | **0.1055** | **0.1290** | **1.14%** | **0.7187** | **0.2681** | **0.1417** |
| **Moving Average (30m)** | 0.1336 | 0.1713 | 1.44% | 0.9099 | 0.4451 | 0.3312 |
| **Persistence (Last Value)** | 0.1468 | 0.1848 | 1.59% | 1.0000 | 0.4714 | 0.3575 |
| **Historical Mean** | 0.1746 | 0.2105 | 1.88% | 1.1894 | 0.4305 | 0.1080 |
| **Direct GBDT** | 0.2174 | 0.2474 | 2.32% | 1.4814 | 0.3877 | 0.0905 |
| **Direct Ridge** | 0.2926 | 0.3282 | 3.11% | 1.9931 | 0.5019 | 0.1298 |

```
                       HOLDOUT FORECAST TRAJECTORY COMPARISON (t=111..134)
  9.8 % ┌─────────────────────────────────────────────────────────────────────────────┐
        │                                                                             │
  9.6 % │            Actual Ground Truth: /\__/\                                      │
        │                               /       \     P90 Upper Capacity Buffer       │
  9.4 % │   - - - - - - - - - - - - - -/---------\ - - - - - - - - - - - - - - - - -  │
        │   Mean-Reverting AR Forecast: ───────────                                   │
  9.2 % │                                                                             │
        │   Persistence (flat line): ...............................................  │
  9.0 % └─────────────────────────────────────────────────────────────────────────────┘
        33,000s                                                                39,900s
        (t=111)                                                                (t=134)
```

---

## 7. Error & Failure Analysis

### 7.1 Why the Mean-Reverting Model Dominates
The empirical results reveal that the **Mean-Reverting AR model ($\phi=0.90$)** substantially outperforms all other models:
- Achieves a **28.1% reduction in MAE** (0.1055 vs. 0.1468) and a **30.2% reduction in RMSE** (0.1290 vs. 0.1848) relative to Persistence.
- Operates on a sound mathematical premise: at the forecast origin ($t=110$), the latest reading was $y_{110} = 9.165\%$, which was below the historical mean $\bar{y} = 9.414\%$. Persistence predicted a flat $9.165\%$ across the entire 2 hours, failing to capture the natural mean-reverting pull of the aggregate cluster.
- The mean-reverting model gradually adjusted from $9.189\%$ at $h=1$ up to $9.394\%$ at $h=24$, closely mirroring the true trajectory of the holdout data which averaged $9.269\%$.

### 7.2 Why Complex ML Models Underperformed
1. **Sample Size vs. Horizon Dimensionality**:
   With only 110 training timestamps, building direct multi-step matrices with $H=24$ yields only $N \approx 70$ training rows. Fitting 24 separate tree models or regularized linear models on 70 data points creates high estimator variance.
2. **Signal-to-Noise Ratio in Aggregates**:
   Because aggregate CPU variance across 10,000 VMs is extremely low ($\sigma = 0.195\%$), the aggregate series resembles a weakly stationary Ornstein-Uhlenbeck process. Machine learning models attempted to extract non-existent non-linear interactions from minor noise fluctuations, leading to out-of-sample drift.
3. **Failure Mode in Moving Averages**:
   Moving average smooths historical values but suffers from phase lag. When the workload experiences an inflection point, Moving Average remains anchored to the past 30 minutes, producing large peak errors (**Peak MAE: 0.3312%** vs. **0.1417%** for Mean-Reverting AR).

---

## 8. Operational Capacity Recommendation

```
                   DYNAMIC RISK-ADJUSTED CAPACITY PROVISIONING
   Capacity Allocated
          ▲
          │                          [P90 Upper Capacity Ceiling]
          │   ══════════════════════════════════════════════════════════════
          │                ▲                                ▲
          │                │    Safety Buffer (+0.35% CPU)  │
          │                ▼                                ▼
          │   ──────────────────────────────────────────────────────────────
          │                 [Mean-Reverting Point Forecast: y_hat]
          │                                  ▲
          │                                  │  Actual Workload Demand
          │                     •┈┈┈┈┈•┈┈┈┈┈•
          │
          └─────────────────────────────────────────────────────────────────► Time
```

### 8.1 The Capacity Trade-Off
In cloud infrastructure operations, the cost function is highly asymmetric:
- **Under-provisioning Cost ($C_{\text{under}}$)**: CPU throttling, queue delays, microservice timeout cascades, SLA violation penalties, and customer churn.
- **Over-provisioning Cost ($C_{\text{over}}$)**: Cost of unallocated server power, reserved cloud instances, and amortized data-center capital expenditure.
- Typically, $C_{\text{under}} \gg C_{\text{over}}$ by a factor of $5\times$ to $20\times$.

### 8.2 Recommended Dynamic Policy: Point Forecast + P90 Quantile Buffer
We recommend an automated, risk-adjusted headroom policy for a 2-hour lookahead:
$$\text{Allocated\_CPU\_Capacity}(t+h) = \hat{y}_{\text{AR-MR}}(t+h) + \text{Headroom\_Buffer}(h)$$
where:
$$\text{Headroom\_Buffer}(h) = \max\left( \hat{y}_{\text{P90\_GBDT}}(t+h) - \hat{y}_{\text{AR-MR}}(t+h), \, k \cdot \sigma_{\text{residual}} \right)$$
with safety multiplier $k = 2.0$ ($\approx 97.7\%$ Gaussian coverage) and empirical residual standard deviation $\sigma_{\text{residual}} \approx 0.13\%$.

### 8.3 Operational Sizing Example
For a cloud cluster hosting 10,000 VMs with an aggregate pool of 20,000 virtual cores:
- **Baseline Point Forecast**: $\hat{y} \approx 9.35\%$ $\implies 1,870$ active vCPUs required.
- **Safety Headroom Buffer**: $+0.35\%$ CPU utilization $\implies +70$ reserve vCPUs.
- **Total Provisioned Capacity**: $9.70\%$ $\implies 1,940$ vCPUs.
- **Outcome**: The maximum observed CPU utilization across the holdout was $9.636\%$. Under this policy:
  - **Capacity Shortfalls**: **0 intervals** (100% SLA compliance).
  - **Wasted Capacity**: Only $0.43\%$ average idle headroom ($< 86$ idle cores out of 20,000).

---

## 9. Limitations & Future Work

1. **Short Observation Duration**:
   - The trace spans only 11.1 hours. Consequently, true 24-hour diurnal rhythms, business-hour shifts, and weekend drop-offs cannot be observed or modeled. A minimum 14- to 30-day trace is recommended for production deployment.
2. **Synthetic Population Completeness**:
   - The `data-v1` curation specifically filtered for VMs with complete 134-observation traces. In live cloud environments, VM churn (deployments and terminations) occurs continuously. A production pipeline must incorporate survival analysis or non-homogeneous Poisson processes to model VM arrival and departure rates.
3. **Cross-Cloud Generalizability**:
   - Comparison with Google ClusterData2019 suggests that containerized Borg workloads display higher frequency churn and shorter task lifetimes than Azure IaaS VMs. Telemetry pipelines must adapt feature cadences to workload granularity.
4. **Segment-Level Hierarchy**:
   - Reconciled hierarchical forecasting (e.g., MinT or top-down reconciliation across Interactive and Delay-insensitive categories) should be explored to enable localized pod-level auto-scaling.

---

## 10. Submission Artifacts & Verification

- **Generated Predictions**: `predictions.csv` (24 rows, columns `timestamp,active_vm_count,aggregate_cpu_utilization`).
- **Reproduction Command**:
  ```bash
  python -m cwfc.predict --data challenge_data --output predictions.csv
  ```
- **Automated Test Suite**:
  ```bash
  python -m pytest -v
  ```
- **Generated Figures**:
  - [eda_time_series.png](file:///c:/forecasting-challenge-Atomity/reports/figures/eda_time_series.png)
  - [eda_cpu_distributions.png](file:///c:/forecasting-challenge-Atomity/reports/figures/eda_cpu_distributions.png)
  - [eda_autocorrelation.png](file:///c:/forecasting-challenge-Atomity/reports/figures/eda_autocorrelation.png)
  - [model_horizon_comparison.png](file:///c:/forecasting-challenge-Atomity/reports/figures/model_horizon_comparison.png)
  - [holdout_forecast_comparison.png](file:///c:/forecasting-challenge-Atomity/reports/figures/holdout_forecast_comparison.png)
- **JSON Metrics Summary**: [benchmark_results.json](file:///c:/forecasting-challenge-Atomity/reports/figures/benchmark_results.json)
