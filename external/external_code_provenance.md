# External code provenance

This repository does not vendor external model or splitter repositories. Full-scale split and prediction outputs are provided in the Zenodo data package and consumed by the scripts here.

| Tool/model | Role | Repository/checkpoint | Version or commit status | Local code changes | Output path |
|---|---|---|---|---|---|
| Chemprop / D-MPNN | GNN baseline | `https://github.com/chemprop/chemprop` | v2.2.1 | None | `data/predictions/dmpnn/` |
| MolCLR | pretrained graph baseline | `https://github.com/yuyangw/MolCLR` | no tagged release in the public repository used; original public code was used | None | `data/predictions/molclr/` |
| ChemBERTa-3 | language-model baseline | `https://github.com/deepforestsci/chemberta3`; checkpoint `DeepChem/ChemBERTa-77M-MLM` | no tagged release in the public repository used; original public code was used | None | `data/predictions/chemberta3/` |
| DataSAIL | leakage-aware splitter and ESA backbone | `https://github.com/kalininalab/DataSAIL` | v1.2.3 | None | `data/splits/datasail/`, `data/splits/esa/` |
| LoHi | splitter comparison | `https://github.com/SteshinSS/lohi_splitter` | no tagged release in the public repository used; original public code was used | None | `data/splits/lohi/` |
| TensorFlow / Keras | FCNN baselines | `tensorflow` Python package | 2.15.0 | None | `data/predictions/fcnn_*` |

Standalone XGBoost and FCNN fingerprint scripts are provided in `scripts/models/` for new split files.

Prediction tables are expected under:

```text
data/predictions/<model>/<benchmark>/<task_group>/<dataset>/fold_<n>/test_predictions.csv
```

Each table should contain `molecules` and one prediction column per target. Supported prediction column names include `<target>_pred`, `pred_<target>`, `<target>_score`, and `prob_<target>`.
