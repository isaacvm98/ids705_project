# IDS 705 Final Project: Retail Demand Forecasting

## What This Repository Is

This repository contains our final IDS 705 project on weekly retail demand forecasting using the M5 Walmart dataset. The project asks three practical questions:

1. Which model family gives the strongest out-of-sample forecast accuracy?
2. How much historical data is needed before forecasts become reliable?
3. How much accuracy can be retained while keeping the model interpretable?

To answer those questions, we compare naive baselines, linear models, tree ensembles, a generalized additive model (GAM), and a neural-network benchmark on the same store-category-week forecasting task. We evaluate accuracy, data sufficiency, runtime, and interpretability rather than treating the problem as a single-metric leaderboard.

The modeling target is **weekly unit sales aggregated to the `store × category × week` level**. This keeps the problem operationally meaningful while making a multi-model comparison feasible within a course project.

## Repository Layout

```text
ids705_project/
├── artifacts/                  # Saved final-model artifacts and metadata
├── clean_data/                 # Generated processed data (created by preprocessing)
├── notebooks/
│   ├── data/                   # Location where the raw M5 files should be placed
│   └── models/                 # Main project notebooks and helper script
├── outputs/                    # Saved model outputs and interpretability plots
├── reports/                    # Final report, memo materials, and report figures
├── requirements.txt            # Python dependencies
└── README.md
```

The main notebooks used in the final project are:

- `01_data_loading_and_eda.ipynb`
- `02_preprocessing_and_feature_engineering.ipynb`
- `03_baseline_and_linear_models.ipynb`
- `05_random_forest.ipynb`
- `06_lightgbm.ipynb`
- `07_gam.ipynb`
- `08_mlp.ipynb`
- `11_data_sufficiency_and_comparison.ipynb`

Additional notebooks in `notebooks/models/` are supporting or iterative analysis notebooks kept for project transparency.

The notebooks include markdown explanations and inline code comments so that each preprocessing, modeling, and evaluation step can be followed directly in the project files.

## Data

This project uses the **M5 Forecasting Accuracy** dataset from Walmart.

### Raw data needed

Place the following files in `notebooks/data/` before running the notebooks:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `sample_submission.csv`

### How to get the data

The raw files are not currently committed in this repository. Download them from the Kaggle M5 competition page:

<https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>

After downloading, create the directory if needed and copy the files there:

```bash
mkdir -p notebooks/data
```

Then place the five CSV files listed above inside `notebooks/data/`.

### What preprocessing does

The preprocessing pipeline:

- loads the item-level daily M5 data
- aggregates daily sales to weekly `store × category` demand
- engineers lag, rolling, calendar, event, SNAP, and price features
- creates train/test tables for the downstream model notebooks

The main generated files are written to `clean_data/`, including:

- `weekly_agg_sales.csv`
- `weekly_features.csv`
- `train.csv`
- `test.csv`

Some later notebooks use the filename `training.csv` instead of `train.csv`. If you are reproducing the full workflow from scratch, create that alias after running Notebook 02:

```bash
cp clean_data/train.csv clean_data/training.csv
```

That keeps the GAM and pooled-model workflows aligned with the preprocessing outputs.

## Environment Setup

We recommend **Python 3.12** and a virtual environment.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

## Reproducing the Project

All final-project notebooks are in `notebooks/models/`.

### Full rerun from raw data

Run the notebooks in this order:

1. `01_data_loading_and_eda.ipynb`
   - loads the raw M5 data
   - aggregates daily item sales to weekly `store × category` demand
   - saves `clean_data/weekly_agg_sales.csv`

2. `02_preprocessing_and_feature_engineering.ipynb`
   - builds lag, rolling, calendar, event, SNAP, and price features
   - creates the processed train/test splits
   - saves `clean_data/train.csv`, `clean_data/test.csv`, and `clean_data/weekly_features.csv`

3. Create the pooled-workflow alias:

   ```bash
   cp clean_data/train.csv clean_data/training.csv
   ```

4. `03_baseline_and_linear_models.ipynb`
   - runs naive baselines and regularized linear models
   - saves the final Ridge artifact to `artifacts/ridge/`

5. `05_random_forest.ipynb`
   - fits and tunes the pooled Random Forest model
   - saves the best artifact to `artifacts/random_forest/`

6. `06_lightgbm.ipynb`
   - fits and tunes the pooled LightGBM model
   - saves the best artifact to `artifacts/lightgbm/`

7. `07_gam.ipynb`
   - fits the GAM benchmark
   - saves GAM outputs and metadata

8. `08_mlp.ipynb`
   - fits the neural-network benchmark
   - saves the final MLP artifact to `artifacts/mlp/`

9. `11_data_sufficiency_and_comparison.ipynb`
   - loads the saved final-model artifacts
   - runs the reduced-history and Pareto-vs-full data sufficiency experiments
   - synthesizes cross-model comparison and interpretability outputs

### Fastest way to inspect final results

If you do not want to retrain every model, you can start once:

- the raw M5 data have been downloaded
- Notebooks 01 and 02 have been run
- `clean_data/training.csv` has been created from `clean_data/train.csv`

At that point you can use the already committed final-model artifacts in:

- `artifacts/ridge/`
- `artifacts/random_forest/`
- `artifacts/lightgbm/`
- `artifacts/gam/`
- `artifacts/mlp/`

and run:

- `11_data_sufficiency_and_comparison.ipynb`

This is the most direct way to reproduce the final comparison phase used in the report.

## What Each Main Notebook Does

### Core workflow

- `01_data_loading_and_eda.ipynb`
  - raw-data loading, weekly aggregation, and exploratory analysis

- `02_preprocessing_and_feature_engineering.ipynb`
  - shared feature engineering and downstream split construction

- `03_baseline_and_linear_models.ipynb`
  - naive baselines, linear models, evaluation, and Ridge artifact saving

- `07_gam.ipynb`
  - GAM benchmark, evaluation, and interpretable output generation

- `08_mlp.ipynb`
  - neural-network benchmark and artifact saving

- `11_data_sufficiency_and_comparison.ipynb`
  - final comparison, data sufficiency, Pareto analysis, and cross-model interpretation

### Tree-model workflow

- `04_tree_data_loader.ipynb`
  - helper notebook for pooled-model loading logic

- `05_random_forest.ipynb`
  - Random Forest fitting and tuning

- `06_lightgbm.ipynb`
  - LightGBM fitting and tuning

- `09_tree_shap_analysis.ipynb`
  - original tree-based SHAP notebook

- `10_tree_model_evaluation.ipynb`
  - pooled tree-model evaluation outputs

- `data_loader_00.py`
  - helper script used by the pooled tree-model notebooks

## Artifacts and Outputs

Final-model notebooks save fitted model objects and metadata into `artifacts/`. Each model folder contains:

- a saved model or model bundle (`.joblib`)
- a metadata file (`*_metadata.json`)

These artifacts are used by the final comparison notebook so that model selection and data sufficiency experiments can be run in a consistent, reproducible way.

Saved report-facing figures and supporting outputs are in:

- `reports/figures/`
- `outputs/`

## Dependencies

The required Python dependencies are listed in `requirements.txt`. The current project uses:

- `jupyter`
- `joblib`
- `lightgbm`
- `matplotlib`
- `numpy`
- `openpyxl`
- `pandas`
- `pygam`
- `scikit-learn`
- `scipy`
- `shap`
- `statsmodels`

### Environment notes

- On macOS, `lightgbm` may require `libomp`. If import fails, install it with Homebrew:

  ```bash
  brew install libomp
  ```

- SHAP must be installed in the same Jupyter kernel that runs the notebook.

## What the Final Results Support

The final report and appendix are based on this codebase. At a high level, the project found:

- Ridge, LightGBM, and Random Forest were the strongest overall forecasting models
- the tuned GAM remained a credible interpretable alternative
- the MLP benchmark did not outperform the stronger tabular methods
- LightGBM was the most stable model under reduced-history data sufficiency tests

## Final Deliverables

This repository contains both the analysis code and the written submission materials:

- code and notebooks: `notebooks/models/`
- saved model artifacts: `artifacts/`
- report materials and report figures: `reports/`

## Notes for Grading / Testing

1. install the environment from `requirements.txt`
2. place the raw M5 files in `notebooks/data/`
3. run Notebooks 01 and 02
4. create `clean_data/training.csv` from `clean_data/train.csv`
5. run any final model notebook or run `11_data_sufficiency_and_comparison.ipynb`

This reflects the actual current project workflow in the repository.
