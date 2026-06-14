# Model scripts

These scripts provide portable fingerprint baselines for split CSV files. The archived prediction tables used in the main analyses are provided separately.

## XGBoost

```bash
python scripts/models/train_xgboost_fingerprint.py \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --target target_column \
  --task regression \
  --output-dir outputs/model_examples/xgboost
```

Use `--task classification` for binary classification datasets.

## FCNN

```bash
python scripts/models/train_fcnn_fingerprint.py \
  --train path/to/train.csv \
  --valid path/to/valid.csv \
  --test path/to/test.csv \
  --target target_column \
  --task regression \
  --output-dir outputs/model_examples/fcnn
```

The FCNN script requires TensorFlow.
