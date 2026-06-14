#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from empty_scaffold.metrics import classification_summary, regression_summary
from empty_scaffold.modeling import predict_xgboost_classification, predict_xgboost_regression


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost Morgan-fingerprint baseline on provided train/test CSV files.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--smiles-column", default="molecules")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-boost-round", type=int, default=500)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "regression":
        pred = predict_xgboost_regression(
            train_df, test_df, args.target,
            smiles_col=args.smiles_column,
            num_boost_round=args.num_boost_round,
        )
        metrics = regression_summary(pred["y_true"], pred["y_pred"])
        out = pred[[args.smiles_column, "y_true", "y_pred"]].rename(columns={"y_true": args.target, "y_pred": f"pred_{args.target}"})
    else:
        pred = predict_xgboost_classification(
            train_df, test_df, args.target,
            smiles_col=args.smiles_column,
            num_boost_round=args.num_boost_round,
        )
        metrics = classification_summary(pred["y_true"], pred["y_score"], pred["y_pred"])
        out = pred[[args.smiles_column, "y_true", "y_score", "y_pred"]].rename(columns={
            "y_true": args.target,
            "y_score": f"prob_{args.target}",
            "y_pred": f"pred_{args.target}",
        })

    out.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame([{**metrics, "target": args.target, "model": "xgboost_fingerprint"}]).to_csv(out_dir / "test_metrics.csv", index=False)
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
