# Empty scaffold allocation bias

Code for **Empty scaffold allocation bias in molecular distribution shift evaluation**.

The repository includes scaffold/no-scaffold diagnostics, split-allocation audits, ESA repair utilities, leakage metrics, model-summary scripts, and executable notebooks.

Processed datasets, split files, and prediction tables are archived at <https://doi.org/10.5281/zenodo.20337102>.

## Installation

```bash
conda env create -f environment.yml
conda activate empty-scaffold
pip install -e .
pytest -q
```

Typical install time on a normal desktop computer is approximately 5–10 minutes, depending on network speed and package download time.

## Notebooks

Run notebooks from the repository root:

```bash
jupyter lab notebooks/01_scaffold_split_from_raw.ipynb
jupyter lab notebooks/02_regression_metric_comparison_xgboost.ipynb
jupyter lab notebooks/03_classification_metric_comparison_xgboost.ipynb
jupyter lab notebooks/04_datasail_esa_repair_audit.ipynb
```

The notebooks use bundled demo data and write results under `outputs/`.

Expected run time for the bundled demo notebooks is approximately 1–5 minutes per notebook on a normal desktop computer. Full-data analyses require the Zenodo data package and take longer.

## Demo data

Bundled examples:

```text
data/demo/scaffold_from_scratch/       molecule tables for scaffold-fold analyses
data/demo/datasail_esa_cases/          raw DataSAIL split examples for ESA repair audit
```

For full analyses, download the Zenodo archive and place or symlink:

```text
data/processed/
data/splits/
data/predictions/
```

## Full-data scripts

```bash
python scripts/analysis/scaffold_split.py --input-root data/processed --output-root data/splits/scaffold
python scripts/analysis/classification_normpr_analysis.py --split-root data/splits/scaffold --prediction-root data/predictions --out-dir results/classification_normpr
python scripts/analysis/regression_relmae_analysis.py --split-root data/splits/scaffold --prediction-root data/predictions --out-dir results/regression_relmae
python scripts/analysis/split_leakage_metrics.py --split-root data/splits/esa/D_train_support_priority --out-dir results/esa_leakage --include-valid
```

See `docs/full_reproduction.md` for the full folder contract.

## Model scripts

Standalone XGBoost and FCNN fingerprint baselines are in `scripts/models/`.

The manuscript-scale analyses use archived prediction tables from the Zenodo data package.

## Repository layout

```text
configs/             dataset and ESA configuration files
data/demo/           bundled examples
docs/                full-data layout notes
external/            external-code provenance
notebooks/           executable walkthroughs
scripts/analysis/    full-data analysis scripts
scripts/models/      XGBoost and FCNN baseline scripts
src/empty_scaffold/  shared Python code
tests/               pytest checks
```

## License

This project is released under the MIT License. See `LICENSE` for details.
