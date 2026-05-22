# Empty-Scaffold Allocation Bias in Molecular Distribution Shift Evaluation

This repository contains the analysis scripts used for the study:

**Empty scaffold allocation bias in molecular distribution shift evaluation**

The analyses examine how molecules with empty Bemis-Murcko scaffold representations affect molecular benchmark splitting and downstream model evaluation. The repository focuses on the numerical workflow: structural-cohesion analysis, split allocation, ESA repair, classification and regression performance analysis, and metric sensitivity analysis.

## Repository contents

```text
empty-scaffold-allocation/
  README.md
  environment.yml
  LICENSE

  configs/
    datasets.yaml
    split_ratios.yaml
    esa_balanced.yaml
    esa_trainsupport.yaml

  data/
    README.md

  scripts/
    01_structural_cohesion_and_distribution_shift.py
    02_scaffold_split.py
    03_umap_cluster_split.py
    04_datasail_and_esa_repair.py
    05_esa_condition_sweep.py
    06_classification_normpr_analysis.py
    07_regression_relmae_analysis.py
    08_metric_sensitivity_analysis.py
```

## Data availability

The original MoleculeNet datasets are publicly available from the MoleculeNet data portal:

- https://moleculenet.org/datasets-1

The original Deep-PK/ADMET datasets are publicly available from the Deep-PK data portal:

- https://biosig.lab.uq.edu.au/deeppk/data

The processed datasets, train/validation/test split files, and model-prediction tables generated in this study are archived on Zenodo:

- https://doi.org/10.5281/zenodo.20337102

The Zenodo archive contains three folders:

```text
processed/
splits/
predictions/
```

After downloading and extracting the archive, place these folders under `data/`:

```text
data/
  processed/
  splits/
  predictions/
```

See `data/README.md` for the expected folder layout.

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate empty-scaffold-allocation
```

## Workflow

The scripts use relative paths by default. Run commands from the repository root.

### 1. Structural cohesion and distribution-shift analysis

```bash
python scripts/01_structural_cohesion_and_distribution_shift.py \
  --input data/processed \
  --out results/part2_structural_shift
```

### 2. Standard Bemis-Murcko scaffold split

```bash
python scripts/02_scaffold_split.py \
  --input-root data/processed \
  --output-root data/splits/scaffold
```

### 3. UMAP clustering split

```bash
python scripts/03_umap_cluster_split.py \
  --input-root data/processed \
  --output-root data/splits/umap
```

### 4. ESA repair from DataSAIL splits

```bash
python scripts/04_datasail_and_esa_repair.py \
  --input-root data/processed \
  --datasail-root data/splits/datasail \
  --output-root data/splits/esa
```

### 5. ESA configuration sweep

```bash
python scripts/05_esa_condition_sweep.py \
  --input-root data/processed \
  --datasail-root data/splits/datasail \
  --out results/esa_condition_sweep
```

### 6. Classification performance analysis

```bash
python scripts/06_classification_normpr_analysis.py \
  --split-root data/splits/scaffold \
  --prediction-root data/predictions \
  --out-dir results/classification_normpr
```

### 7. Regression performance analysis

```bash
python scripts/07_regression_relmae_analysis.py \
  --split-root data/splits/scaffold \
  --prediction-root data/predictions \
  --out-dir results/regression_relmae
```

### 8. Metric sensitivity analysis

```bash
python scripts/08_metric_sensitivity_analysis.py \
  --classification-eval results/classification_normpr/classification_eval_long.csv \
  --regression-eval results/regression_relmae/regression_eval_long.csv \
  --output results/metric_sensitivity/metric_sensitivity_summary.xlsx
```

## Data layout expected by scripts

```text
data/
  processed/
    admet/
      no_scaffold_ratio_ranking.csv
      classification/*.csv
      regression/*.csv
    moleculenet/
      classification/*.csv
      regression/*.csv

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

The `processed/` folder contains the datasets retained for analysis. The `splits/` folder contains train/validation/test assignments. The `predictions/` folder contains model-prediction tables used for the classification, regression, and metric-sensitivity analyses.

## Notes on external splitters

DataSAIL and LoHi splits used in the study are included in the archived `splits/` folder. The scripts in this repository use those split files directly. Regenerating DataSAIL or LoHi outputs from scratch may require separate splitter-specific environments and the corresponding external packages.

## Code availability statement

The custom code used for scaffold extraction, no-scaffold annotation, split generation, no-scaffold allocation diagnostics, ESA repair, metric calculation, and numerical summary generation is available in this repository. Processed datasets, split files, and model-prediction tables are available at Zenodo: https://doi.org/10.5281/zenodo.20337102.
