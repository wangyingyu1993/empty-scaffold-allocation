#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classification performance analysis on no-scaffold test subsets.

Input layout:
  data/splits/scaffold/
    <benchmark>/classification/<dataset>/fold_<n>/train.csv
    <benchmark>/classification/<dataset>/fold_<n>/valid.csv
    <benchmark>/classification/<dataset>/fold_<n>/test.csv

  data/predictions/
    <model>/<benchmark>/classification/<dataset>/fold_<n>/test_predictions.csv

Output layout:
  results/classification_normpr/
    classification_eval_long.csv
    classification_summary_tables.csv
    classification_analysis_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, roc_auc_score

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

MIN_POSITIVES = 10
MIN_NEGATIVES = 10


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


def to_binary_label(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    unique = set(sorted(values.dropna().unique().tolist()))
    if not unique:
        return pd.Series([np.nan] * len(series), index=series.index, dtype=float)
    if unique.issubset({0, 1}):
        return values.astype(float)
    if unique.issubset({-1, 1}) or unique.issubset({-1, 0, 1}):
        return (values > 0).astype(float)
    if values.dropna().min() >= 0 and values.dropna().max() <= 1:
        return (values >= 0.5).astype(float)
    return pd.Series([np.nan] * len(series), index=series.index, dtype=float)


def norm_pr(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0:
        return np.nan

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return np.nan

    positive_rate = float(np.mean(y_true))
    ap = average_precision_score(y_true, y_score)

    if positive_rate >= 1:
        return np.nan

    return float((ap - positive_rate) / (1 - positive_rate))


def safe_ap(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def is_eligible_binary_set(y_true) -> bool:
    y_true = np.asarray(y_true, dtype=float)
    y_true = y_true[np.isfinite(y_true)]
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    return positives >= MIN_POSITIVES and negatives >= MIN_NEGATIVES


def collect_dataset_dirs(split_root: Path):
    records = []

    for benchmark in ["admet", "moleculenet"]:
        base = split_root / benchmark / "classification"
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
    base = prediction_root / model / benchmark / "classification" / dataset / fold
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


def score_column_for_target(pred_df: pd.DataFrame, target: str):
    candidates = [
        f"{target}_pred",
        f"{target}_score",
        f"{target}_prob",
        f"pred_{target}",
        f"score_{target}",
        f"prob_{target}",
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


def evaluate_subset(df: pd.DataFrame, target: str, score_col: str, subset_name: str):
    sub = df.copy()
    y_true = to_binary_label(sub[target])
    y_score = pd.to_numeric(sub[score_col], errors="coerce")

    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask].astype(float)
    y_score = y_score[mask].astype(float)

    if not is_eligible_binary_set(y_true):
        return None

    return {
        "Subset": subset_name,
        "N": int(len(y_true)),
        "N_Positive": int(np.sum(y_true == 1)),
        "N_Negative": int(np.sum(y_true == 0)),
        "Positive_Rate": float(np.mean(y_true)),
        "AP": safe_ap(y_true, y_score),
        "ROC_AUC": safe_auc(y_true, y_score),
        "normPR": norm_pr(y_true, y_score),
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
                score_col = score_column_for_target(merged, target)
                if score_col is None or score_col == target:
                    continue

                for subset_name, subset_df in [
                    ("all_test", merged),
                    ("no_scaffold", merged[merged["ScaffoldStatus"] == "No_Scaffold"]),
                    ("scaffold_bearing", merged[merged["ScaffoldStatus"] == "Has_Scaffold"]),
                ]:
                    metrics = evaluate_subset(subset_df, target, score_col, subset_name)
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
    out["Reference_Type"] = "raw"

    no_rows = out[out["Subset"] == "no_scaffold"].copy()
    ordinary_rows = out[out["Subset"] == "all_test"].copy()

    ordinary_ref = ordinary_rows.groupby(
        ["Benchmark", "Dataset", "Target", "Model"], as_index=False
    )["normPR"].median().rename(columns={"normPR": "Ordinary_Median_normPR"})

    out = out.merge(ordinary_ref, on=["Benchmark", "Dataset", "Target", "Model"], how="left")
    out["Delta_normPR_vs_Ordinary"] = out["normPR"] - out["Ordinary_Median_normPR"]

    within = out.pivot_table(
        index=["Benchmark", "Dataset", "Target", "Model", "Fold"],
        columns="Subset",
        values="normPR",
        aggfunc="first",
    ).reset_index()

    if "no_scaffold" in within.columns and "scaffold_bearing" in within.columns:
        within["NoMinusScaffoldBearing_normPR"] = within["no_scaffold"] - within["scaffold_bearing"]
        out = out.merge(
            within[["Benchmark", "Dataset", "Target", "Model", "Fold", "NoMinusScaffoldBearing_normPR"]],
            on=["Benchmark", "Dataset", "Target", "Model", "Fold"],
            how="left",
        )
    else:
        out["NoMinusScaffoldBearing_normPR"] = np.nan

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
            "Median_AP": float(sub["AP"].median()),
            "Median_ROC_AUC": float(sub["ROC_AUC"].median()),
            "Median_normPR": float(sub["normPR"].median()),
            "NearBaseline_normPR_le_0p05": float((sub["normPR"] <= 0.05).mean()),
            "NearBaseline_normPR_le_0p02": float((sub["normPR"] <= 0.02).mean()),
        })

    no_sub = eval_long[eval_long["Subset"] == "no_scaffold"].copy()
    if "Delta_normPR_vs_Ordinary" in no_sub.columns and not no_sub.empty:
        for benchmark, group in no_sub.groupby("Benchmark"):
            rows.append({
                "Table": "no_scaffold_delta_summary",
                "Group": benchmark,
                "N_EvaluationPoints": int(len(group)),
                "Median_Delta_normPR_vs_Ordinary": float(group["Delta_normPR_vs_Ordinary"].median()),
                "Fraction_Below_Ordinary": float((group["Delta_normPR_vs_Ordinary"] < 0).mean()),
                "Median_WithinFold_NoMinusScaffoldBearing": float(group["NoMinusScaffoldBearing_normPR"].median()),
                "Fraction_Below_ScaffoldBearing": float((group["NoMinusScaffoldBearing_normPR"] < 0).mean()),
            })

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze classification performance on no-scaffold test subsets.")
    parser.add_argument("--split-root", type=Path, default=Path("data/splits/scaffold"), help="Scaffold split directory.")
    parser.add_argument("--prediction-root", type=Path, default=Path("data/predictions"), help="Model prediction directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/classification_normpr"), help="Output directory.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model folders under data/predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_records = collect_dataset_dirs(args.split_root)
    if dataset_records.empty:
        raise SystemExit(f"No classification split directories found under {args.split_root}")

    rows = []
    for _, record in dataset_records.iterrows():
        rows.extend(process_dataset(record, args.prediction_root, args.models))
        print(f"Analyzed classification dataset: {record['Benchmark']}/{record['Dataset']}")

    eval_long = pd.DataFrame(rows)
    eval_long = add_reference_comparisons(eval_long)
    summary = build_summary_tables(eval_long)

    eval_long.to_csv(args.out_dir / "classification_eval_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "classification_summary_tables.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "split_root": str(args.split_root),
        "prediction_root": str(args.prediction_root),
        "output_directory": str(args.out_dir),
        "models": args.models,
        "minimum_positive_samples": MIN_POSITIVES,
        "minimum_negative_samples": MIN_NEGATIVES,
    }
    (args.out_dir / "classification_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Evaluation points: {len(eval_long)}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
