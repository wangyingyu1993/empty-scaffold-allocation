#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric sensitivity analysis for primary classification and regression metrics.

Input layout:
  results/classification_normpr/classification_eval_long.csv
  results/regression_relmae/regression_eval_long.csv

Output layout:
  results/metric_sensitivity/
    metric_sensitivity_summary.xlsx
    metric_sensitivity_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NEAR_BASELINE_NORMPR_THRESHOLDS = [0.05, 0.02]
RELMAE_INFLATION_THRESHOLDS = [2.0, 3.0]


def safe_spearman(x, y) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    return float(spearmanr(x[mask], y[mask]).correlation)


def direction_agreement(a, b, threshold: float = 0.0) -> float:
    a = pd.to_numeric(pd.Series(a), errors="coerce")
    b = pd.to_numeric(pd.Series(b), errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return np.nan
    return float((a[mask] > threshold).eq(b[mask] > threshold).mean())


def load_table(path: Path, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def classification_metric_summary(class_df: pd.DataFrame) -> pd.DataFrame:
    if class_df.empty:
        return pd.DataFrame()

    no_df = class_df[class_df["Subset"] == "no_scaffold"].copy()
    ref_df = class_df[class_df["Subset"] == "all_test"].copy()

    rows = []
    for metric in ["AP", "ROC_AUC", "normPR"]:
        if metric not in class_df.columns:
            continue

        ordinary_median = ref_df[metric].median()
        no_median = no_df[metric].median()

        if metric == "normPR":
            degradation = no_df["Delta_normPR_vs_Ordinary"] if "Delta_normPR_vs_Ordinary" in no_df.columns else np.nan
            degradation_rate = float((degradation < 0).mean()) if not isinstance(degradation, float) else np.nan
            agreement = np.nan
            corr = np.nan
        else:
            ref = ref_df.groupby(
                ["Benchmark", "Dataset", "Target", "Model"], as_index=False
            )[metric].median().rename(columns={metric: f"Ordinary_Median_{metric}"})

            merged = no_df.merge(ref, on=["Benchmark", "Dataset", "Target", "Model"], how="left")
            degradation = merged[metric] - merged[f"Ordinary_Median_{metric}"]
            degradation_rate = float((degradation < 0).mean())

            if "Delta_normPR_vs_Ordinary" in merged.columns:
                corr = safe_spearman(-degradation, -merged["Delta_normPR_vs_Ordinary"])
                agreement = direction_agreement(-degradation, -merged["Delta_normPR_vs_Ordinary"])
            else:
                corr = np.nan
                agreement = np.nan

        rows.append({
            "Metric": metric,
            "Ordinary_Median": float(ordinary_median) if pd.notna(ordinary_median) else np.nan,
            "NoScaffold_Median": float(no_median) if pd.notna(no_median) else np.nan,
            "Median_Change_NoMinusOrdinary": float(no_median - ordinary_median) if pd.notna(no_median) and pd.notna(ordinary_median) else np.nan,
            "Degradation_Rate": degradation_rate,
            "Spearman_With_normPR_Degradation": corr,
            "Direction_Agreement_With_normPR_Degradation": agreement,
        })

    return pd.DataFrame(rows)


def classification_near_baseline_summary(class_df: pd.DataFrame) -> pd.DataFrame:
    if class_df.empty or "normPR" not in class_df.columns:
        return pd.DataFrame()

    rows = []
    for subset in ["all_test", "no_scaffold", "scaffold_bearing"]:
        sub = class_df[class_df["Subset"] == subset].copy()
        if sub.empty:
            continue

        row = {
            "Subset": subset,
            "N": int(len(sub)),
            "Median_AP": float(sub["AP"].median()) if "AP" in sub.columns else np.nan,
            "Median_ROC_AUC": float(sub["ROC_AUC"].median()) if "ROC_AUC" in sub.columns else np.nan,
            "Median_normPR": float(sub["normPR"].median()),
            "Median_Positive_Rate": float(sub["Positive_Rate"].median()) if "Positive_Rate" in sub.columns else np.nan,
        }

        for threshold in NEAR_BASELINE_NORMPR_THRESHOLDS:
            row[f"normPR_le_{str(threshold).replace('.', 'p')}"] = float((sub["normPR"] <= threshold).mean())

        if "ROC_AUC" in sub.columns:
            row["ROC_AUC_gt_0p55"] = float((sub["ROC_AUC"] > 0.55).mean())

        rows.append(row)

    return pd.DataFrame(rows)


def classification_metric_discordance_cases(class_df: pd.DataFrame, n_cases: int = 20) -> pd.DataFrame:
    if class_df.empty:
        return pd.DataFrame()

    sub = class_df[class_df["Subset"] == "no_scaffold"].copy()
    required = {"AP", "ROC_AUC", "normPR", "Positive_Rate"}
    if not required.issubset(set(sub.columns)):
        return pd.DataFrame()

    sub["near_baseline_normPR"] = sub["normPR"].abs()
    sub["roc_minus_normpr_rank_gap"] = sub["ROC_AUC"] - sub["normPR"]
    sub["ap_minus_prior_gap"] = sub["AP"] - sub["Positive_Rate"]

    cases = []

    c1 = sub.sort_values(["Positive_Rate", "AP", "near_baseline_normPR"], ascending=[False, False, True]).head(n_cases // 4)
    c1 = c1.assign(Case_Type="high_AP_high_prior_low_normPR")
    cases.append(c1)

    c2 = sub.sort_values(["near_baseline_normPR", "AP"], ascending=[True, False]).head(n_cases // 4)
    c2 = c2.assign(Case_Type="nonzero_AP_near_prior_normPR")
    cases.append(c2)

    c3 = sub[sub["ROC_AUC"] > 0.55].sort_values(["near_baseline_normPR", "ROC_AUC"], ascending=[True, False]).head(n_cases // 2)
    c3 = c3.assign(Case_Type="ROC_AUC_above_random_near_prior_normPR")
    cases.append(c3)

    out = pd.concat(cases, axis=0).drop_duplicates(
        subset=["Benchmark", "Dataset", "Target", "Model", "Fold", "Subset"]
    )

    columns = [
        "Case_Type", "Benchmark", "Dataset", "Target", "Model", "Fold",
        "N", "N_Positive", "N_Negative", "Positive_Rate", "AP", "ROC_AUC", "normPR",
        "Delta_normPR_vs_Ordinary", "NoMinusScaffoldBearing_normPR",
    ]
    return out[[c for c in columns if c in out.columns]].head(n_cases)


def regression_metric_summary(reg_df: pd.DataFrame) -> pd.DataFrame:
    if reg_df.empty:
        return pd.DataFrame()

    no_df = reg_df[reg_df["Subset"] == "no_scaffold"].copy()
    ref_df = reg_df[reg_df["Subset"] == "all_test"].copy()

    rows = []

    error_metrics = ["MAE", "RMSE", "relMAE"]
    association_metrics = ["R2", "Pearson_r", "Spearman_rho"]

    for metric in error_metrics:
        if metric not in reg_df.columns:
            continue

        ordinary_median = ref_df[metric].median()
        no_median = no_df[metric].median()

        ref = ref_df.groupby(
            ["Benchmark", "Dataset", "Target", "Model"], as_index=False
        )[metric].median().rename(columns={metric: f"Ordinary_Median_{metric}"})

        merged = no_df.merge(ref, on=["Benchmark", "Dataset", "Target", "Model"], how="left")
        ratio = merged[metric] / merged[f"Ordinary_Median_{metric}"]

        if "relMAE_Inflation" in merged.columns:
            corr = safe_spearman(ratio, merged["relMAE_Inflation"])
            agreement = direction_agreement(ratio, merged["relMAE_Inflation"], threshold=1.0)
        else:
            corr = np.nan
            agreement = np.nan

        rows.append({
            "Metric": metric,
            "Metric_Type": "error",
            "Ordinary_Median": float(ordinary_median) if pd.notna(ordinary_median) else np.nan,
            "NoScaffold_Median": float(no_median) if pd.notna(no_median) else np.nan,
            "Pooled_NoScaffold_to_Ordinary_Ratio": float(no_median / ordinary_median) if ordinary_median else np.nan,
            "Median_ModelTask_Ratio_or_Drop": float(ratio.median()),
            "Degradation_Rate": float((ratio > 1).mean()),
            "Spearman_With_relMAE_Inflation": corr,
            "Direction_Agreement_With_relMAE_Inflation": agreement,
        })

    for metric in association_metrics:
        if metric not in reg_df.columns:
            continue

        ordinary_median = ref_df[metric].median()
        no_median = no_df[metric].median()

        ref = ref_df.groupby(
            ["Benchmark", "Dataset", "Target", "Model"], as_index=False
        )[metric].median().rename(columns={metric: f"Ordinary_Median_{metric}"})

        merged = no_df.merge(ref, on=["Benchmark", "Dataset", "Target", "Model"], how="left")
        drop = merged[f"Ordinary_Median_{metric}"] - merged[metric]

        if "relMAE_Inflation" in merged.columns:
            corr = safe_spearman(drop, merged["relMAE_Inflation"])
            agreement = direction_agreement(drop, merged["relMAE_Inflation"] - 1.0)
        else:
            corr = np.nan
            agreement = np.nan

        rows.append({
            "Metric": metric,
            "Metric_Type": "association",
            "Ordinary_Median": float(ordinary_median) if pd.notna(ordinary_median) else np.nan,
            "NoScaffold_Median": float(no_median) if pd.notna(no_median) else np.nan,
            "Pooled_NoScaffold_to_Ordinary_Ratio": np.nan,
            "Median_ModelTask_Ratio_or_Drop": float(drop.median()),
            "Degradation_Rate": float((drop > 0).mean()),
            "Spearman_With_relMAE_Inflation": corr,
            "Direction_Agreement_With_relMAE_Inflation": agreement,
        })

    return pd.DataFrame(rows)


def regression_tail_enrichment(reg_df: pd.DataFrame) -> pd.DataFrame:
    if reg_df.empty or "relMAE_Inflation" not in reg_df.columns:
        return pd.DataFrame()

    no_df = reg_df[reg_df["Subset"] == "no_scaffold"].copy()
    rows = []

    for benchmark, group in no_df.groupby("Benchmark"):
        for threshold in RELMAE_INFLATION_THRESHOLDS:
            rows.append({
                "Benchmark": benchmark,
                "Metric": "relMAE",
                "Tail_Criterion": f">={threshold:g}x",
                "NoScaffold_Tail_Probability": float((group["relMAE_Inflation"] >= threshold).mean()),
                "N_EvaluationPoints": int(len(group)),
            })

    for threshold in RELMAE_INFLATION_THRESHOLDS:
        rows.append({
            "Benchmark": "pooled",
            "Metric": "relMAE",
            "Tail_Criterion": f">={threshold:g}x",
            "NoScaffold_Tail_Probability": float((no_df["relMAE_Inflation"] >= threshold).mean()),
            "N_EvaluationPoints": int(len(no_df)),
        })

    return pd.DataFrame(rows)


def regression_metric_discordance_cases(reg_df: pd.DataFrame, n_cases: int = 20) -> pd.DataFrame:
    if reg_df.empty:
        return pd.DataFrame()

    sub = reg_df[reg_df["Subset"] == "no_scaffold"].copy()
    required = {"MAE_Ratio", "TargetScale_Ratio", "relMAE_Inflation", "R2", "Pearson_r", "RMSE", "MAE"}
    if not required.issubset(set(sub.columns)):
        return pd.DataFrame()

    cases = []

    c1 = sub.sort_values(["relMAE_Inflation", "MAE_Ratio"], ascending=[False, True]).head(n_cases // 4)
    c1 = c1.assign(Case_Type="high_relMAE_from_target_scale")
    cases.append(c1)

    c2 = sub.sort_values(["relMAE_Inflation", "R2"], ascending=[False, False]).head(n_cases // 4)
    c2 = c2.assign(Case_Type="high_relMAE_high_R2")
    cases.append(c2)

    c3 = sub.sort_values(["relMAE_Inflation", "Pearson_r"], ascending=[False, False]).head(n_cases // 4)
    c3 = c3.assign(Case_Type="high_relMAE_high_Pearson")
    cases.append(c3)

    sub["RMSE_to_MAE"] = sub["RMSE"] / sub["MAE"]
    c4 = sub.sort_values(["RMSE_to_MAE", "relMAE_Inflation"], ascending=[False, False]).head(n_cases // 4)
    c4 = c4.assign(Case_Type="RMSE_tail_sensitivity")
    cases.append(c4)

    out = pd.concat(cases, axis=0).drop_duplicates(
        subset=["Benchmark", "Dataset", "Target", "Model", "Fold", "Subset"]
    )

    columns = [
        "Case_Type", "Benchmark", "Dataset", "Target", "Model", "Fold",
        "N", "MAE", "RMSE", "relMAE", "MAE_Ratio", "TargetScale_Ratio",
        "relMAE_Inflation", "R2", "Pearson_r", "Spearman_rho",
    ]
    return out[[c for c in columns if c in out.columns]].head(n_cases)


def write_workbook(output: Path, sheets: dict[str, pd.DataFrame]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, table in sheets.items():
            safe_name = name[:31]
            if table is None or table.empty:
                pd.DataFrame({"message": ["No data available"]}).to_excel(writer, sheet_name=safe_name, index=False)
            else:
                table.to_excel(writer, sheet_name=safe_name, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run metric sensitivity analysis.")
    parser.add_argument(
        "--classification-eval",
        type=Path,
        default=Path("results/classification_normpr/classification_eval_long.csv"),
        help="Classification evaluation table.",
    )
    parser.add_argument(
        "--regression-eval",
        type=Path,
        default=Path("results/regression_relmae/regression_eval_long.csv"),
        help="Regression evaluation table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metric_sensitivity/metric_sensitivity_summary.xlsx"),
        help="Output Excel workbook.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classification = load_table(args.classification_eval, required=False)
    regression = load_table(args.regression_eval, required=False)

    sheets = {
        "classification_metrics": classification_metric_summary(classification),
        "classification_near_base": classification_near_baseline_summary(classification),
        "classification_cases": classification_metric_discordance_cases(classification),
        "regression_metrics": regression_metric_summary(regression),
        "regression_tail": regression_tail_enrichment(regression),
        "regression_cases": regression_metric_discordance_cases(regression),
    }

    write_workbook(args.output, sheets)

    metadata = {
        "classification_eval": str(args.classification_eval),
        "regression_eval": str(args.regression_eval),
        "output": str(args.output),
        "near_baseline_normpr_thresholds": NEAR_BASELINE_NORMPR_THRESHOLDS,
        "relmae_inflation_thresholds": RELMAE_INFLATION_THRESHOLDS,
    }
    metadata_path = args.output.parent / "metric_sensitivity_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Output workbook: {args.output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
