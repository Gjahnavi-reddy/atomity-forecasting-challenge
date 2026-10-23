# Cloud Workload Forecasting Challenge

AI/ML take-home challenge built from the **Microsoft Azure Public Dataset V2** VM trace.

## Start here

1. Read [`ASSIGNMENT.md`](ASSIGNMENT.md).
2. Read [`DATASET.md`](DATASET.md).
3. Download the **`data-v1` GitHub Release** dataset and extract it as `challenge_data/`.
4. Read [`SUBMISSION.md`](SUBMISSION.md).
5. Implement your analysis and forecasting pipeline.

The challenge dataset is distributed as a versioned GitHub Release rather than committed to the repository. The `data-v1` release contains the fixed curated dataset used for this assignment.

## Environment setup

Python **3.11 or newer** is required.

Create a virtual environment:

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python -m pytest -q
```

The repository includes a small test suite covering the starter forecasting utilities. Run it before beginning the assignment.

## Dataset download

The hiring dataset is distributed as the fixed `data-v1` release of this repository.

**Download:** [Azure Dataset v1 Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

Download `data-v1.tar.gz` from the release and extract it into the repository as `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

The resulting layout is:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Verify the dataset before starting the assignment:

```bash
cd challenge_data
sha256sum -c SHA256SUMS
```

You should see:

```text
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

The release is a real, curated subset of the Microsoft Azure Public Dataset V2. It is not synthetic data.

The dataset contains 10,000 VMs with 134 CPU measurements per VM at five-minute intervals, for a total of 1,340,000 CPU observations.

Use the exact `data-v1` release provided for this assignment. Do not substitute another dataset or a different version of the Azure Public Dataset.

## What you are solving

The underlying data contains anonymized VM workload telemetry sampled every five minutes. Your job is to understand the workload and build a defensible forecasting pipeline for near-term aggregate demand.

The challenge is intentionally open-ended. We evaluate reasoning, time-series methodology, leakage prevention, error analysis, reproducibility, and the quality of your operational conclusions—not just model complexity.

## Expected environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Model dependencies are intentionally not prescribed. Add only what you need and document them.

## Source

Microsoft Azure Public Dataset V2:
https://github.com/Azure/AzurePublicDataset

The official V2 documentation describes a 30-day VM trace with five-minute CPU readings, VM metadata, and encrypted subscription/deployment/VM identifiers. The full upstream trace is much larger than this challenge release.

## Running the Forecasting Pipeline & Generating Predictions

To train the model on the 110-observation training window and generate the official `predictions.csv` for the 24-step (2-hour) holdout horizon:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

Optional CLI flags:
- `--data`: Path to dataset directory (default: `challenge_data`).
- `--output`: Path to write predictions CSV (default: `predictions.csv`).
- `--model`: Model selection (`mean_reverting`, `ridge`, `gbm`, `persistence`, `moving_average`). Defaults to `mean_reverting`.

### Running Tests

Run the comprehensive test suite (including validation, leakage prevention, and `predictions.csv` contract tests):

```bash
python -m pytest -v
```

### Reproducing Analysis & Benchmarks

To re-run the exploratory workload analysis and generate report figures:
```bash
python scripts/generate_eda_artifacts.py
```

To re-run the 5-fold expanding-window cross-validation benchmark across all models:
```bash
python scripts/benchmark_models.py
```

An interactive exploratory notebook is also available at:
[`notebooks/01_exploratory_workload_analysis.ipynb`](notebooks/01_exploratory_workload_analysis.ipynb)

## Main Modeling Decisions & Summary

1. **Target Definitions**:
   - `active_vm_count`: Strict count of active VMs with valid readings at timestamp $t$. Filtered population exhibits zero churn across the trace, remaining constant at 10,000 VMs.
   - `aggregate_cpu_utilization`: Mean of VM-level `avg_cpu` readings at timestamp $t$ (mean: 9.39% ± 0.19%).
2. **Leakage-Free Temporal Validation**:
   - Random splitting is strictly avoided. We use an expanding-window rolling-origin evaluation (5 folds, horizon = 24 steps = 2 hours) across the first 110 observations.
   - Metadata summary statistics (`max_cpu`, `avg_cpu`, `p95_max_cpu`) are excluded from feature inputs as they aggregate the full 30-day trace.
3. **Core Model: Mean-Reverting Autoregressive ($\text{AR-MR}$, $\phi=0.90$)**:
   - Autocorrelation decays monotonically from $\rho_1 = 0.67$ to $\rho_{24} \approx 0$.
   - The mean-reverting model ($\hat{y}_{t+h} = \mu + \phi^h (y_t - \mu)$) captures short-term persistence while decaying gracefully toward the unconditional mean.
   - Achieves the lowest cross-validation MAE (**0.2138%**) and best holdout MAE (**0.1055%**, a **28.1% improvement** over persistence) and RMSE (**0.1290%**, a **30.2% improvement** over persistence).
4. **Operational Capacity Policy**:
   - Dynamic Headroom: Provision at point forecast $\hat{y}_{t+h}$ plus an upper risk buffer derived from empirical GBDT P90 quantile estimation (+0.35% CPU / +35 cores per 10k vCPUs).
   - Achieves zero SLA violations / capacity shortfalls across the holdout horizon with $< 4.5\%$ idle buffer.

For the full 6-page technical report and deep dive, see:
[`reports/final_report.md`](reports/final_report.md).

## Submission

Submit your work as a repository or archive according to [`SUBMISSION.md`](SUBMISSION.md).
