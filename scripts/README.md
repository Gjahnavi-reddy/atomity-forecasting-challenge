# Scripts

Helper scripts for benchmarking models, generating exploratory plots, and building project assets.

## Available scripts

- `benchmark_models.py`: Runs 5-fold temporal cross-validation on the 110 training steps across all models, evaluates against the holdout set, and writes metrics and figures to `reports/figures/`.
  ```bash
  python scripts/benchmark_models.py
  ```
- `generate_eda_artifacts.py`: Computes dataset summary statistics and generates the EDA charts saved in `reports/figures/`.
  ```bash
  python scripts/generate_eda_artifacts.py
  ```
- `create_notebook.py`: Generates the analysis notebook in `notebooks/`.
  ```bash
  python scripts/create_notebook.py
  ```
