# Full-data layout

Download the large files from <https://doi.org/10.5281/zenodo.20337102> and place or symlink these folders in the repository root.

```text
data/processed/
data/splits/
data/predictions/
```

Expected split layout:

```text
data/splits/scaffold/<benchmark>/<task_group>/<dataset>/fold_<n>/{train,valid,test}.csv
data/splits/umap/<benchmark>/<task_group>/<dataset>/fold_<n>/{train,valid,test}.csv
data/splits/datasail/<benchmark>/<task_group>/<dataset>/{train,valid,test}.csv
data/splits/lohi/<benchmark>/<task_group>/<dataset>/{train,valid,test}.csv
data/splits/esa/<condition>/<benchmark>/<task_group>/<dataset>/{train,valid,test}.csv
```

Expected prediction layout:

```text
data/predictions/<model>/<benchmark>/<task_group>/<dataset>/fold_<n>/test_predictions.csv
```

Model folder names used by the analysis scripts:

```text
chemberta3
dmpnn
fcnn_character
fcnn_fingerprint
molclr
xgboost_character
xgboost_fingerprint
```

Common commands:

```bash
python scripts/analysis/structural_cohesion_and_distribution_shift.py --input data/processed --out results/structural_shift
python scripts/analysis/scaffold_split.py --input-root data/processed --output-root data/splits/scaffold
python scripts/analysis/umap_cluster_split.py --input-root data/processed --output-root data/splits/umap
python scripts/analysis/classification_normpr_analysis.py --split-root data/splits/scaffold --prediction-root data/predictions --out-dir results/classification_normpr
python scripts/analysis/regression_relmae_analysis.py --split-root data/splits/scaffold --prediction-root data/predictions --out-dir results/regression_relmae
python scripts/analysis/metric_sensitivity_analysis.py --classification-eval results/classification_normpr/classification_eval_long.csv --regression-eval results/regression_relmae/regression_eval_long.csv --output results/metric_sensitivity/metric_sensitivity_summary.xlsx
python scripts/analysis/split_leakage_metrics.py --split-root data/splits/esa/D_train_support_priority --out-dir results/esa_leakage --include-valid
```

ESA split generation:

```bash
python scripts/analysis/datasail_and_esa_repair.py --input-root data/processed --output-root data/splits/esa --conditions A_balanced_baseline D_train_support_priority
```
