#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression relMAE analysis on no-scaffold test subsets.

Input layout:
  data/splits/scaffold/
    <benchmark>/regression/<dataset>/fold_<n>/train.csv
    <benchmark>/regression/<dataset>/fold_<n>/valid.csv
    <benchmark>/regression/<dataset>/fold_<n>/test.csv

  data/predictions/
    <model>/<benchmark>/regression/<dataset>/fold_<n>/test_predictions.csv

Output layout:
  results/regression_relmae/
    regression_eval_long.csv
    regression_summary_tables.csv
    regression_analysis_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")

DEFAULT_MODELS = [
    "chemberta3",
    "dmpnn",
    "fcnn_character",
    "fcnn_fingerprint",
    "molclr",
    "xgboost_character",
    "xgboost_fingerprint",
]

MIN_TEST_SAMPLES = 5


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "Invalid_SMILES"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "No_Scaffold"
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return "Error"


def target_columns(df: pd.DataFrame):
    excluded = {"molecules", "Scaffold", "ScaffoldStatus"}
    return [c for c in df.columns if c not in excluded]


def collect_dataset_dirs(split_root: Path):
    records = []

    for benchmark in ["admet", "moleculenet"]:
        base = split_root / benchmark / "regression"
        if not base.exists():
            print(f"[WARN] Missing folder: {base}")
            continue

        for dataset_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            records.append({
                "Benchmark": benchmark,
                "Dataset": normalize_name(dataset_dir.name),
                "DatasetDir": dataset_dir,
            })

    return pd.DataFrame(records)


def read_test_split(dataset_dir: Path, fold: str) -> pd.DataFrame:
    path = dataset_dir / fold / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing test split: {path}")

    df = pd.read_csv(path)
    if "molecules" not in df.columns:
        raise ValueError(f"{path} does not contain a molecules column")

    df = df.copy()
    df["molecules"] = df["molecules"].astype(str)
    df["Scaffold"] = df["molecules"].map(get_scaffold)
    df["ScaffoldStatus"] = np.where(df["Scaffold"] == "No_Scaffold", "No_Scaffold", "Has_Scaffold")
    return df


def prediction_candidates(prediction_root: Path, model: str, benchmark: str, dataset: str, fold: str):
    base = prediction_root / model / benchmark / "regression" / dataset / fold
    return [
        base / "test_predictions.csv",
        base / "predictions.csv",
        base / "test.csv",
    ]


def read_predictions(prediction_root: Path, model: str, benchmark: str, dataset: str, fold: str):
    for path in prediction_candidates(prediction_root, model, benchmark, dataset, fold):
        if path.exists():
            return pd.read_csv(path), path
    return None, None


def prediction_column_for_target(pred_df: pd.DataFrame, target: str):
    candidates = [
        f"{target}_pred",
        f"{target}_prediction",
        f"pred_{target}",
        f"prediction_{target}",
        target,
    ]

    for col in candidates:
        if col in pred_df.columns:
            return col

    numeric_cols = [
        c for c in pred_df.columns
        if c not in {"molecules", "smiles", "id"} and pd.api.types.is_numeric_dtype(pred_df[c])
    ]

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    return None


def align_predictions(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    pred = pred_df.copy()

    if "molecules" in pred.columns:
        pred["molecules"] = pred["molecules"].astype(str)
        return test_df.merge(pred, on="molecules", how="left", suffixes=("", "_predfile"))

    if "smiles" in pred.columns:
        pred = pred.rename(columns={"smiles": "molecules"})
        pred["molecules"] = pred["molecules"].astype(str)
        return test_df.merge(pred, on="molecules", how="left", suffixes=("", "_predfile"))

    if len(pred) != len(test_df):
        raise ValueError("Prediction file has no molecules column and does not match test split length")

    merged = test_df.reset_index(drop=True).copy()
    pred = pred.reset_index(drop=True)
    for col in pred.columns:
        if col not in merged.columns:
            merged[col] = pred[col]
    return merged


def relmae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    scale = float(np.mean(np.abs(y_true)))
    if scale == 0:
        return np.nan

    return float(mean_absolute_error(y_true, y_pred) / scale)


def safe_pearson(y_true, y_pred) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def safe_spearman(y_true, y_pred) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(spearmanr(y_true, y_pred).correlation)


def evaluate_subset(df: pd.DataFrame, target: str, pred_col: str, subset_name: str):
    y_true = pd.to_numeric(df[target], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")

    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask].astype(float).values
    y_pred = y_pred[mask].astype(float).values

    if len(y_true) < MIN_TEST_SAMPLES:
        return None

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    target_scale = float(np.mean(np.abs(y_true)))

    return {
        "Subset": subset_name,
        "N": int(len(y_true)),
        "Target_Scale": target_scale,
        "MAE": mae,
        "RMSE": rmse,
        "relMAE": relmae(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "Pearson_r": safe_pearson(y_true, y_pred),
        "Spearman_rho": safe_spearman(y_true, y_pred),
        "Prediction_SD": float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else np.nan,
        "Target_SD": float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan,
    }


def process_dataset(record, prediction_root: Path, models):
    benchmark = record["Benchmark"]
    dataset = record["Dataset"]
    dataset_dir = Path(record["DatasetDir"])

    folds = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    rows = []

    for fold in folds:
        test_df = read_test_split(dataset_dir, fold)
        targets = target_columns(test_df)

        for model in models:
            pred_df, pred_path = read_predictions(prediction_root, model, benchmark, dataset, fold)
            if pred_df is None:
                continue

            merged = align_predictions(test_df, pred_df)

            for target in targets:
                pred_col = prediction_column_for_target(merged, target)
                if pred_col is None or pred_col == target:
                    continue

                for subset_name, subset_df in [
                    ("all_test", merged),
                    ("no_scaffold", merged[merged["ScaffoldStatus"] == "No_Scaffold"]),
                    ("scaffold_bearing", merged[merged["ScaffoldStatus"] == "Has_Scaffold"]),
                ]:
                    metrics = evaluate_subset(subset_df, target, pred_col, subset_name)
                    if metrics is None:
                        continue

                    rows.append({
                        "Benchmark": benchmark,
                        "Dataset": dataset,
                        "Target": target,
                        "Model": model,
                        "Fold": fold,
                        "Prediction_File": str(pred_path),
                        **metrics,
                    })

    return rows


def add_reference_comparisons(eval_long: pd.DataFrame) -> pd.DataFrame:
    if eval_long.empty:
        return eval_long

    out = eval_long.copy()

    ordinary_ref = out[out["Subset"] == "all_test"].groupby(
        ["Benchmark", "Dataset", "Target", "Model"], as_index=False
    ).agg(
        Ordinary_Median_relMAE=("relMAE", "median"),
        Ordinary_Median_MAE=("MAE", "median"),
        Ordinary_Median_TargetScale=("Target_Scale", "median"),
    )

    out = out.merge(ordinary_ref, on=["Benchmark", "Dataset", "Target", "Model"], how="left")
    out["relMAE_Inflation"] = out["relMAE"] / out["Ordinary_Median_relMAE"]
    out["MAE_Ratio"] = out["MAE"] / out["Ordinary_Median_MAE"]
    out["TargetScale_Ratio"] = out["Target_Scale"] / out["Ordinary_Median_TargetScale"]

    within = out.pivot_table(
        index=["Benchmark", "Dataset", "Target", "Model", "Fold"],
        columns="Subset",
        values="relMAE",
        aggfunc="first",
    ).reset_index()

    if "no_scaffold" in within.columns and "scaffold_bearing" in within.columns:
        within["NoVsScaffoldBearing_relMAE_Ratio"] = within["no_scaffold"] / within["scaffold_bearing"]
        out = out.merge(
            within[["Benchmark", "Dataset", "Target", "Model", "Fold", "NoVsScaffoldBearing_relMAE_Ratio"]],
            on=["Benchmark", "Dataset", "Target", "Model", "Fold"],
            how="left",
        )
    else:
        out["NoVsScaffoldBearing_relMAE_Ratio"] = np.nan

    return out


def build_summary_tables(eval_long: pd.DataFrame) -> pd.DataFrame:
    if eval_long.empty:
        return pd.DataFrame()

    rows = []

    for subset in ["all_test", "no_scaffold", "scaffold_bearing"]:
        sub = eval_long[eval_long["Subset"] == subset]
        if sub.empty:
            continue
        rows.append({
            "Table": "subset_metric_summary",
            "Group": subset,
            "N_EvaluationPoints": int(len(sub)),
            "Median_MAE": float(sub["MAE"].median()),
            "Median_RMSE": float(sub["RMSE"].median()),
            "Median_relMAE": float(sub["relMAE"].median()),
            "Median_R2": float(sub["R2"].median()),
            "Median_Pearson_r": float(sub["Pearson_r"].median()),
            "Median_Spearman_rho": float(sub["Spearman_rho"].median()),
        })

    no_sub = eval_long[eval_long["Subset"] == "no_scaffold"].copy()
    if not no_sub.empty:
        for benchmark, group in no_sub.groupby("Benchmark"):
            rows.append({
                "Table": "no_scaffold_inflation_summary",
                "Group": benchmark,
                "N_EvaluationPoints": int(len(group)),
                "Median_relMAE_Inflation": float(group["relMAE_Inflation"].median()),
                "Fraction_relMAE_Inflation_ge_2x": float((group["relMAE_Inflation"] >= 2).mean()),
                "Fraction_relMAE_Inflation_ge_3x": float((group["relMAE_Inflation"] >= 3).mean()),
                "Median_MAE_Ratio": float(group["MAE_Ratio"].median()),
                "Median_TargetScale_Ratio": float(group["TargetScale_Ratio"].median()),
            })

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze regression relMAE on no-scaffold test subsets.")
    parser.add_argument("--split-root", type=Path, default=Path("data/splits/scaffold"), help="Scaffold split directory.")
    parser.add_argument("--prediction-root", type=Path, default=Path("data/predictions"), help="Model prediction directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/regression_relmae"), help="Output directory.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model folders under data/predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_records = collect_dataset_dirs(args.split_root)
    if dataset_records.empty:
        raise SystemExit(f"No regression split directories found under {args.split_root}")

    rows = []
    for _, record in dataset_records.iterrows():
        rows.extend(process_dataset(record, args.prediction_root, args.models))
        print(f"Analyzed regression dataset: {record['Benchmark']}/{record['Dataset']}")

    eval_long = pd.DataFrame(rows)
    eval_long = add_reference_comparisons(eval_long)
    summary = build_summary_tables(eval_long)

    eval_long.to_csv(args.out_dir / "regression_eval_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "regression_summary_tables.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "split_root": str(args.split_root),
        "prediction_root": str(args.prediction_root),
        "output_directory": str(args.out_dir),
        "models": args.models,
        "minimum_test_samples": MIN_TEST_SAMPLES,
    }
    (args.out_dir / "regression_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Evaluation points: {len(eval_long)}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
