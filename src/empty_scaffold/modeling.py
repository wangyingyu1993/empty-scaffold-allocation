from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from .chem_utils import morgan_fp_array

Task = Literal["regression", "classification"]


def featurize_table(
    df: pd.DataFrame,
    target: str,
    *,
    smiles_col: str = "molecules",
    task: Task,
    n_bits: int = 1024,
):
    rows = []
    features = []
    labels = []
    for _, row in df.iterrows():
        if pd.isna(row.get(target)):
            continue
        fp = morgan_fp_array(row[smiles_col], n_bits=n_bits)
        if fp is None:
            continue
        y = row[target]
        if task == "classification":
            try:
                y = int(y)
            except Exception:
                continue
            if y not in (0, 1):
                continue
        else:
            try:
                y = float(y)
            except Exception:
                continue
        features.append(fp)
        labels.append(y)
        rows.append(row)
    if not features:
        raise ValueError(f"no usable rows for target={target}")
    y_dtype = int if task == "classification" else float
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=y_dtype), pd.DataFrame(rows).reset_index(drop=True)


def predict_xgboost_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    *,
    smiles_col: str = "molecules",
    n_bits: int = 1024,
    num_boost_round: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    x_train, y_train, _ = featurize_table(train_df, target, smiles_col=smiles_col, task="regression", n_bits=n_bits)
    x_test, y_test, test_rows = featurize_table(test_df, target, smiles_col=smiles_col, task="regression", n_bits=n_bits)

    scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = scaler.transform(y_train.reshape(-1, 1)).ravel()
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "seed": seed,
            "nthread": 1,
        },
        xgb.DMatrix(x_train, label=y_train_s),
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    pred_s = booster.predict(xgb.DMatrix(x_test))
    pred = scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    out = test_rows.copy()
    out["y_true"] = y_test
    out["y_pred"] = pred
    return out


def predict_xgboost_classification(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    *,
    smiles_col: str = "molecules",
    n_bits: int = 1024,
    num_boost_round: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    x_train, y_train, _ = featurize_table(train_df, target, smiles_col=smiles_col, task="classification", n_bits=n_bits)
    x_test, y_test, test_rows = featurize_table(test_df, target, smiles_col=smiles_col, task="classification", n_bits=n_bits)
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"training set has a single class for target={target}")

    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "scale_pos_weight": neg / pos,
            "seed": seed,
            "nthread": 1,
        },
        xgb.DMatrix(x_train, label=y_train),
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    prob = booster.predict(xgb.DMatrix(x_test))
    out = test_rows.copy()
    out["y_true"] = y_test
    out["y_score"] = prob
    out["y_pred"] = (prob >= 0.5).astype(int)
    return out
