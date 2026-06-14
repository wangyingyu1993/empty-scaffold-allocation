# DataSAIL and ESA demo cases

This folder contains molecule-level raw DataSAIL split examples and selected DataSAIL+ESA summary rows.

```text
raw_datasail_splits/
  ames/{train,valid,test}.csv
  esol/{train,valid,test}.csv
  pkb/{train,valid,test}.csv

no_scaffold_lookup.csv
datasail_esa_case_summary.csv
datasail_esa_case_proportions_long.csv
```

`raw_datasail_splits/` is audited directly by `scripts/demo/summarize_datasail_esa_cases.py` and `scripts/analysis/split_leakage_metrics.py`. The paired DataSAIL+ESA values are source-table extracts for the same cases. Regenerating all ESA splits requires the full benchmark inputs from Zenodo and the external DataSAIL setup.
