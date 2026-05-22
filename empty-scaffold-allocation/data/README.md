# Data directory

This directory is used for data files required to reproduce the analyses in the manuscript.

Large data files are not stored in the Git repository. The processed datasets, split files, and model-prediction tables generated in this study are archived on Zenodo:

- Zenodo record: https://zenodo.org/records/20337102
- DOI: https://doi.org/10.5281/zenodo.20337102

## Original data sources

The original public datasets should be downloaded from:

- MoleculeNet: https://moleculenet.org/datasets-1
- Deep-PK/ADMET: https://biosig.lab.uq.edu.au/deeppk/data

## Archived data files

The Zenodo archive contains three compressed folders:

```text
processed/
predictions/
splits/
```

These folders correspond to the data layout expected by the analysis scripts.

## Expected local layout

After downloading and extracting the Zenodo archive, the local directory should look like this:

```text
data/
  processed/
    admet/
      no_scaffold_ratio_ranking.csv
      classification/
      regression/
    moleculenet/
      classification/
      regression/

  splits/
    scaffold/
    umap/
    datasail/
    lohi/
    esa/
      A_balanced_baseline/
      D_train_support_priority/

  predictions/
    chemberta3/
    dmpnn/
    fcnn_character/
    fcnn_fingerprint/
    molclr/
    xgboost_character/
    xgboost_fingerprint/
```

## Folder descriptions

### `processed/`

This folder contains the processed datasets used for no-scaffold prevalence analysis, structural-cohesion analysis, distribution-shift analysis, and split generation.

The datasets under `processed/` have already been filtered to retain datasets/tasks with a no-scaffold proportion of at least 0.08 for the analyses requiring this threshold.

Expected structure:

```text
processed/
  admet/
    no_scaffold_ratio_ranking.csv
    classification/*.csv
    regression/*.csv
  moleculenet/
    classification/*.csv
    regression/*.csv
```

Each dataset file is expected to contain a `molecules` column with SMILES strings and one or more target columns.

### `splits/`

This folder contains train/validation/test split files used in the split-composition and model-performance analyses.

Expected structure:

```text
splits/
  scaffold/<benchmark>/<task_group>/<dataset>/fold_<n>/train.csv
  scaffold/<benchmark>/<task_group>/<dataset>/fold_<n>/valid.csv
  scaffold/<benchmark>/<task_group>/<dataset>/fold_<n>/test.csv

  umap/<benchmark>/<task_group>/<dataset>/fold_<n>/train.csv
  umap/<benchmark>/<task_group>/<dataset>/fold_<n>/valid.csv
  umap/<benchmark>/<task_group>/<dataset>/fold_<n>/test.csv

  datasail/<benchmark>/<task_group>/<dataset>/train.csv
  datasail/<benchmark>/<task_group>/<dataset>/valid.csv
  datasail/<benchmark>/<task_group>/<dataset>/test.csv

  lohi/<benchmark>/<task_group>/<dataset>/train.csv
  lohi/<benchmark>/<task_group>/<dataset>/valid.csv
  lohi/<benchmark>/<task_group>/<dataset>/test.csv

  esa/<condition>/<benchmark>/<task_group>/<dataset>/train.csv
  esa/<condition>/<benchmark>/<task_group>/<dataset>/valid.csv
  esa/<condition>/<benchmark>/<task_group>/<dataset>/test.csv
```

Here, `<benchmark>` is `admet` or `moleculenet`, and `<task_group>` is `classification` or `regression`.

### `predictions/`

This folder contains model-prediction tables used for classification and regression performance analyses.

Expected structure:

```text
predictions/
  <model>/<benchmark>/classification/<dataset>/fold_<n>/test_predictions.csv
  <model>/<benchmark>/regression/<dataset>/fold_<n>/test_predictions.csv
```

The expected model folders are:

```text
chemberta3/
dmpnn/
fcnn_character/
fcnn_fingerprint/
molclr/
xgboost_character/
xgboost_fingerprint/
```

Prediction files should contain molecule identifiers or SMILES strings and prediction columns matching the corresponding target names.

## Use with analysis scripts

The scripts in this repository assume the following default paths:

```text
data/processed
data/splits
data/predictions
```

For example:

```bash
python scripts/02_structural_cohesion_and_distribution_shift.py \
  --input data/processed \
  --out results/part2_structural_shift

python scripts/03_scaffold_split.py \
  --input-root data/processed \
  --output-root data/splits/scaffold

python scripts/07_classification_normpr_analysis.py \
  --split-root data/splits/scaffold \
  --prediction-root data/predictions \
  --out-dir results/classification_normpr

python scripts/08_regression_relmae_analysis.py \
  --split-root data/splits/scaffold \
  --prediction-root data/predictions \
  --out-dir results/regression_relmae
```

## Notes

The Git repository contains the analysis scripts and lightweight configuration files. The Zenodo archive contains the processed datasets, split files, and prediction tables required to reproduce the numerical analyses.
