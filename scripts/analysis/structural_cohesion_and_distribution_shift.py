#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural cohesion and distribution-shift analysis for no-scaffold molecules.

Input layout:
  data/processed/
    admet/classification/*.csv
    admet/regression/*.csv
    moleculenet/classification/*.csv
    moleculenet/regression/*.csv

The processed directory is expected to contain the datasets retained for analysis.
All CSV files found under the four benchmark/task folders are analyzed.

Outputs:
  dataset_inventory_0p08.csv
  Table_S4_structural_metrics_dataset_level.csv
  Table_S5_label_shift_task_level.csv
  part2_summary_values.json
  part2_auto_summary.md
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import warnings

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")

# =========================================================
# 0. Paths / config
# =========================================================
INPUT_DIRS = {}
OUT_DIR = Path("results/part2_structural_shift")

RANDOM_SEED = 20250328
FP_RADIUS = 2
FP_NBITS = 2048

EXACT_PAIRWISE_N = 60000
MAX_RANDOM_PAIRS = 60000
COHESION_NEIGHBORS = 60000
NN_CANDIDATE_SAMPLE = 60000
MIN_GROUP_SIZE_FOR_METRIC = 5
MIN_TASK_GROUP_SIZE = 6


def configure_paths(input_dir: Path, out_dir: Path):
    global INPUT_DIRS, OUT_DIR
    INPUT_DIRS = {
        ("admet", "classification"): input_dir / "admet" / "classification",
        ("admet", "regression"): input_dir / "admet" / "regression",
        ("moleculenet", "classification"): input_dir / "moleculenet" / "classification",
        ("moleculenet", "regression"): input_dir / "moleculenet" / "regression",
    }
    OUT_DIR = out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Basic helpers
# =========================================================
def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def pretty_name(name: str) -> str:
    return str(name).replace("_", " ")


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


def make_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    except Exception:
        return None


def gini(values):
    arr = np.asarray([float(x) for x in values if pd.notna(x)], dtype=float)
    if arr.size == 0:
        return np.nan
    if np.allclose(arr, 0):
        return 0.0
    if np.min(arr) < 0:
        arr = arr - np.min(arr)
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def js_divergence_bernoulli(p, q, eps=1e-12):
    p = min(max(float(p), eps), 1 - eps)
    q = min(max(float(q), eps), 1 - eps)
    m = 0.5 * (p + q)

    def kl_bern(a, b):
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))

    return 0.5 * kl_bern(p, m) + 0.5 * kl_bern(q, m)


def standardized_mean_diff(x, y):
    x = np.asarray([v for v in x if pd.notna(v)], dtype=float)
    y = np.asarray([v for v in y if pd.notna(v)], dtype=float)
    if len(x) < 2 or len(y) < 2:
        return np.nan

    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    pooled = ((len(x) - 1) * var_x + (len(y) - 1) * var_y) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return np.nan
    d = (mean_x - mean_y) / math.sqrt(pooled)

    # Hedges correction
    correction = 1 - (3 / (4 * (len(x) + len(y)) - 9))
    return float(d * correction)


def ks_statistic(x, y):
    x = np.sort(np.asarray([v for v in x if pd.notna(v)], dtype=float))
    y = np.sort(np.asarray([v for v in y if pd.notna(v)], dtype=float))
    if len(x) == 0 or len(y) == 0:
        return np.nan
    grid = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def to_binary_label(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce")
    uniq = set(sorted(pd.Series(vals).dropna().unique().tolist()))
    if len(uniq) == 0:
        return pd.Series(dtype=float)

    if uniq.issubset({0, 1}):
        return vals.astype(float)
    if uniq.issubset({-1, 1}) or uniq.issubset({-1, 0, 1}):
        return (vals > 0).astype(float)

    valid = vals.dropna()
    if len(valid) > 0 and valid.min() >= 0 and valid.max() <= 1:
        # fallback for probability-like binaries
        return (vals >= 0.5).astype(float)

    return pd.Series([np.nan] * len(series), index=series.index, dtype=float)


# =========================================================
# 2. Similarity helpers (sampling-aware)
# =========================================================
def _rng(dataset_name: str):
    digest = hashlib.sha256(f"{RANDOM_SEED}:{dataset_name}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:8], 16))


def sample_pairs(n, max_pairs, rng):
    seen = set()
    out = []
    max_possible = n * (n - 1) // 2
    target = min(max_pairs, max_possible)
    while len(out) < target:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if (i, j) not in seen:
            seen.add((i, j))
            out.append((i, j))
    return out


def pairwise_mean_similarity(fps, dataset_name="default"):
    fps = [fp for fp in fps if fp is not None]
    n = len(fps)
    if n < 2:
        return np.nan

    rng = _rng(dataset_name + "_pair")
    sims = []

    if n <= EXACT_PAIRWISE_N:
        for i in range(n):
            row = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            sims.extend(row)
    else:
        for i, j in sample_pairs(n, MAX_RANDOM_PAIRS, rng):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    if len(sims) == 0:
        return np.nan
    return float(np.mean(sims))


def molecule_cohesion_scores(fps, dataset_name="default"):
    fps = [fp for fp in fps if fp is not None]
    n = len(fps)
    if n < 2:
        return []

    rng = _rng(dataset_name + "_cohesion")
    scores = []
    for i, fp in enumerate(fps):
        candidates = [j for j in range(n) if j != i]
        if len(candidates) == 0:
            continue
        if len(candidates) > COHESION_NEIGHBORS:
            candidates = rng.sample(candidates, COHESION_NEIGHBORS)
        sims = [DataStructs.TanimotoSimilarity(fp, fps[j]) for j in candidates]
        if len(sims) > 0:
            scores.append(float(np.mean(sims)))
    return scores


def max_similarity_to_candidates(fp, candidate_fps, rng, sample_limit=NN_CANDIDATE_SAMPLE):
    if fp is None or len(candidate_fps) == 0:
        return np.nan

    candidates = candidate_fps
    if len(candidates) > sample_limit:
        idx = rng.sample(range(len(candidates)), sample_limit)
        candidates = [candidates[k] for k in idx]

    sims = DataStructs.BulkTanimotoSimilarity(fp, candidates)
    if len(sims) == 0:
        return np.nan
    return float(np.max(sims))


# =========================================================
# 3. Data loading / preprocessing
# =========================================================
def collect_dataset_files():
    records = []
    for (benchmark, task_group), folder in INPUT_DIRS.items():
        if not folder.exists():
            print(f"[WARN] missing folder: {folder}")
            continue
        csvs = sorted(folder.glob("*.csv"))
        print(f"[INFO] {benchmark}/{task_group}: found {len(csvs)} csv files")
        for csv_path in csvs:
            records.append({
                "Benchmark": benchmark,
                "TaskGroup": task_group,
                "Dataset": normalize_name(csv_path.stem),
                "CSV_Path": csv_path,
            })
    return pd.DataFrame(records)


def read_and_annotate_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "molecules" not in df.columns:
        raise ValueError(f"{csv_path} missing molecules column")

    work = df.copy()
    work["molecules"] = work["molecules"].astype(str)
    work["Scaffold"] = work["molecules"].map(get_scaffold)
    work["Fingerprint"] = work["molecules"].map(make_fingerprint)

    work = work[
        (~work["Scaffold"].isin(["Invalid_SMILES", "Error"])) &
        work["Fingerprint"].notna()
    ].copy()

    work["ScaffoldStatus"] = np.where(
        work["Scaffold"] == "No_Scaffold", "No_Scaffold", "Has_Scaffold"
    )
    return work


# =========================================================
# 4. Dataset inventory
# =========================================================
def build_dataset_inventory(dataset_files):
    rows = []
    data_cache = {}

    for _, row in dataset_files.iterrows():
        benchmark = row["Benchmark"]
        task_group = row["TaskGroup"]
        dataset = row["Dataset"]
        csv_path = row["CSV_Path"]

        df = read_and_annotate_dataset(csv_path)
        data_cache[(benchmark, task_group, dataset)] = df

        n_total = len(df)
        n_no = int((df["ScaffoldStatus"] == "No_Scaffold").sum())
        n_has = int((df["ScaffoldStatus"] == "Has_Scaffold").sum())
        prop_no = n_no / n_total if n_total > 0 else np.nan

        rows.append({
            "Benchmark": benchmark,
            "TaskGroup": task_group,
            "Dataset": dataset,
            "CSV_Path": str(csv_path),
            "Total_Molecules": n_total,
            "No_Scaffold_Molecules": n_no,
            "Has_Scaffold_Molecules": n_has,
            "No_Scaffold_Proportion": prop_no,
            "Threshold_OK_ge_0p08": bool(prop_no >= 0.08) if pd.notna(prop_no) else False,
        })

    inventory = pd.DataFrame(rows).sort_values(
        ["Benchmark", "TaskGroup", "Dataset"]
    ).reset_index(drop=True)

    inventory.to_csv(OUT_DIR / "dataset_inventory_0p08.csv", index=False, encoding="utf-8-sig")
    return inventory, data_cache


# =========================================================
# 5. Structural metrics
# =========================================================
def compute_structural_metrics_for_dataset(df, dataset_name):
    no_df = df[df["ScaffoldStatus"] == "No_Scaffold"].copy()
    has_df = df[df["ScaffoldStatus"] == "Has_Scaffold"].copy()

    no_fps = no_df["Fingerprint"].tolist()
    has_fps = has_df["Fingerprint"].tolist()

    result = {
        "No_Group_N": len(no_df),
        "Has_Group_N": len(has_df),
        "No_PairwiseMean": np.nan,
        "HasMixed_PairwiseMean": np.nan,
        "TrueScaffoldClass_PairwiseMean": np.nan,
        "No_CohesionGini": np.nan,
        "HasMixed_CohesionGini": np.nan,
        "TrueScaffoldClass_CohesionGini": np.nan,
        "SameScaffold_NNMean": np.nan,
        "CrossScaffold_NNMean": np.nan,
        "NoToScaffold_NNMean": np.nan,
        "NoToNo_NNMean": np.nan,
        "ScaffoldClassCount_ge2": 0,
    }

    # Pairwise mean and cohesion gini
    if len(no_fps) >= MIN_GROUP_SIZE_FOR_METRIC:
        result["No_PairwiseMean"] = pairwise_mean_similarity(no_fps, dataset_name + "_no")
        result["No_CohesionGini"] = gini(molecule_cohesion_scores(no_fps, dataset_name + "_no"))

    if len(has_fps) >= MIN_GROUP_SIZE_FOR_METRIC:
        result["HasMixed_PairwiseMean"] = pairwise_mean_similarity(has_fps, dataset_name + "_has")
        result["HasMixed_CohesionGini"] = gini(molecule_cohesion_scores(has_fps, dataset_name + "_has"))

    # True scaffold class metrics: average across real scaffold groups (size-weighted)
    class_pair_means = []
    class_ginis = []
    class_sizes = []

    scaffold_groups = []
    for scaffold, sub in has_df.groupby("Scaffold"):
        if scaffold in ["No_Scaffold", "Invalid_SMILES", "Error"]:
            continue
        if len(sub) >= 2:
            scaffold_groups.append((scaffold, sub.copy()))

    result["ScaffoldClassCount_ge2"] = len(scaffold_groups)

    for scaffold, sub in scaffold_groups:
        fps = sub["Fingerprint"].tolist()
        pair_mean = pairwise_mean_similarity(fps, f"{dataset_name}_{scaffold}_pair")
        cg = gini(molecule_cohesion_scores(fps, f"{dataset_name}_{scaffold}_cohesion"))
        if pd.notna(pair_mean):
            class_pair_means.append(pair_mean)
            class_sizes.append(len(sub))
        if pd.notna(cg):
            class_ginis.append((cg, len(sub)))

    if len(class_pair_means) > 0:
        result["TrueScaffoldClass_PairwiseMean"] = float(
            np.average(class_pair_means, weights=class_sizes[:len(class_pair_means)])
        )
    if len(class_ginis) > 0:
        result["TrueScaffoldClass_CohesionGini"] = float(
            np.average([x for x, _ in class_ginis], weights=[w for _, w in class_ginis])
        )

    # Nearest-neighbor relations
    rng = _rng(dataset_name + "_nn")

    # same scaffold NN
    same_nn_scores = []
    cross_nn_scores = []
    no_to_scaf_scores = []
    no_to_no_scores = []

    # precompute scaffold-bearing fps grouped by scaffold
    scaffold_to_fps = {
        scaffold: sub["Fingerprint"].tolist()
        for scaffold, sub in has_df.groupby("Scaffold")
        if scaffold not in ["No_Scaffold", "Invalid_SMILES", "Error"]
    }
    all_has_scaffolds = list(scaffold_to_fps.keys())

    for scaffold, sub in has_df.groupby("Scaffold"):
        if scaffold in ["No_Scaffold", "Invalid_SMILES", "Error"]:
            continue
        this_fps = sub["Fingerprint"].tolist()
        other_fps = []
        for other_scaf, fps in scaffold_to_fps.items():
            if other_scaf != scaffold:
                other_fps.extend(fps)

        for idx, fp in enumerate(this_fps):
            # same scaffold NN
            same_candidates = this_fps[:idx] + this_fps[idx + 1:]
            if len(same_candidates) > 0:
                same_nn_scores.append(max_similarity_to_candidates(fp, same_candidates, rng))

            # cross scaffold NN
            if len(other_fps) > 0:
                cross_nn_scores.append(max_similarity_to_candidates(fp, other_fps, rng))

    # no scaffold -> scaffold NN
    for fp in no_fps:
        if len(has_fps) > 0:
            no_to_scaf_scores.append(max_similarity_to_candidates(fp, has_fps, rng))
        other_no = [x for x in no_fps if x is not fp]
        if len(other_no) > 0:
            no_to_no_scores.append(max_similarity_to_candidates(fp, other_no, rng))

    if len(same_nn_scores) > 0:
        result["SameScaffold_NNMean"] = float(np.nanmean(same_nn_scores))
    if len(cross_nn_scores) > 0:
        result["CrossScaffold_NNMean"] = float(np.nanmean(cross_nn_scores))
    if len(no_to_scaf_scores) > 0:
        result["NoToScaffold_NNMean"] = float(np.nanmean(no_to_scaf_scores))
    if len(no_to_no_scores) > 0:
        result["NoToNo_NNMean"] = float(np.nanmean(no_to_no_scores))

    return result


def build_structural_metrics(dataset_inventory, data_cache):
    rows = []
    for _, row in dataset_inventory.iterrows():
        benchmark = row["Benchmark"]
        task_group = row["TaskGroup"]
        dataset = row["Dataset"]

        df = data_cache[(benchmark, task_group, dataset)]
        metrics = compute_structural_metrics_for_dataset(df, dataset)

        out = {
            "Benchmark": benchmark,
            "TaskGroup": task_group,
            "Dataset": dataset,
            "Total_Molecules": row["Total_Molecules"],
            "No_Scaffold_Molecules": row["No_Scaffold_Molecules"],
            "Has_Scaffold_Molecules": row["Has_Scaffold_Molecules"],
            "No_Scaffold_Proportion": row["No_Scaffold_Proportion"],
        }
        out.update(metrics)
        rows.append(out)

    structural = pd.DataFrame(rows).sort_values(
        ["Benchmark", "TaskGroup", "Dataset"]
    ).reset_index(drop=True)
    structural.to_csv(
        OUT_DIR / "Table_S4_structural_metrics_dataset_level.csv",
        index=False,
        encoding="utf-8-sig"
    )
    return structural


# =========================================================
# 6. Label-shift metrics
# =========================================================
def label_columns(df):
    return [c for c in df.columns if c not in ["molecules", "Scaffold", "Fingerprint", "ScaffoldStatus"]]


def compute_label_shift_for_target(df, task_group, target_col):
    sub = df[["ScaffoldStatus", target_col]].copy()
    sub = sub[sub[target_col].notna()].copy()

    no_mask = sub["ScaffoldStatus"] == "No_Scaffold"
    has_mask = sub["ScaffoldStatus"] == "Has_Scaffold"

    if task_group == "regression":
        values = pd.to_numeric(sub[target_col], errors="coerce")
        sub[target_col] = values
        sub = sub[sub[target_col].notna()].copy()

        x = sub.loc[sub["ScaffoldStatus"] == "No_Scaffold", target_col].values.astype(float)
        y = sub.loc[sub["ScaffoldStatus"] == "Has_Scaffold", target_col].values.astype(float)

        if len(x) < MIN_TASK_GROUP_SIZE or len(y) < MIN_TASK_GROUP_SIZE:
            return None

        return {
            "Target": target_col,
            "N_NoScaffold": len(x),
            "N_HasScaffold": len(y),
            "No_Mean": float(np.mean(x)),
            "Has_Mean": float(np.mean(y)),
            "No_SD": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "Has_SD": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
            "Mean_Diff_NoMinusHas": float(np.mean(x) - np.mean(y)),
            "Abs_StandardizedMeanDiff": abs(standardized_mean_diff(x, y)),
            "KS_Statistic": ks_statistic(x, y),
            "Metric_1_Name": "Abs_StandardizedMeanDiff",
            "Metric_1_Value": abs(standardized_mean_diff(x, y)),
            "Metric_2_Name": "KS_Statistic",
            "Metric_2_Value": ks_statistic(x, y),
        }

    if task_group == "classification":
        binary = to_binary_label(sub[target_col])
        sub[target_col] = binary
        sub = sub[sub[target_col].notna()].copy()

        x = sub.loc[sub["ScaffoldStatus"] == "No_Scaffold", target_col].values.astype(float)
        y = sub.loc[sub["ScaffoldStatus"] == "Has_Scaffold", target_col].values.astype(float)

        if len(x) < MIN_TASK_GROUP_SIZE or len(y) < MIN_TASK_GROUP_SIZE:
            return None

        p_no = float(np.mean(x))
        p_has = float(np.mean(y))
        return {
            "Target": target_col,
            "N_NoScaffold": len(x),
            "N_HasScaffold": len(y),
            "No_PositiveRate": p_no,
            "Has_PositiveRate": p_has,
            "PositiveRate_Diff_NoMinusHas": p_no - p_has,
            "Abs_PositiveRate_Diff": abs(p_no - p_has),
            "Bernoulli_JSD": js_divergence_bernoulli(p_no, p_has),
            "Metric_1_Name": "Abs_PositiveRate_Diff",
            "Metric_1_Value": abs(p_no - p_has),
            "Metric_2_Name": "Bernoulli_JSD",
            "Metric_2_Value": js_divergence_bernoulli(p_no, p_has),
        }

    return None


def build_label_shift_metrics(dataset_inventory, data_cache):
    rows = []

    for _, row in dataset_inventory.iterrows():
        benchmark = row["Benchmark"]
        task_group = row["TaskGroup"]
        dataset = row["Dataset"]

        df = data_cache[(benchmark, task_group, dataset)]
        target_cols = label_columns(df)

        for target in target_cols:
            result = compute_label_shift_for_target(df, task_group, target)
            if result is None:
                continue

            out = {
                "Benchmark": benchmark,
                "TaskGroup": task_group,
                "Dataset": dataset,
                "Target": result.pop("Target"),
                "Total_Molecules": row["Total_Molecules"],
                "No_Scaffold_Molecules": row["No_Scaffold_Molecules"],
                "Has_Scaffold_Molecules": row["Has_Scaffold_Molecules"],
                "No_Scaffold_Proportion": row["No_Scaffold_Proportion"],
            }
            out.update(result)
            rows.append(out)

    label_df = pd.DataFrame(rows)
    if label_df.empty:
        label_df = pd.DataFrame(columns=[
            "Benchmark", "TaskGroup", "Dataset", "Target",
            "Total_Molecules", "No_Scaffold_Molecules", "Has_Scaffold_Molecules",
            "No_Scaffold_Proportion"
        ])
    else:
        label_df = label_df.sort_values(
            ["TaskGroup", "Benchmark", "Dataset", "Target"]
        ).reset_index(drop=True)

    label_df.to_csv(
        OUT_DIR / "Table_S5_label_shift_task_level.csv",
        index=False,
        encoding="utf-8-sig"
    )
    return label_df


# =========================================================
# 9. Summary values for writing
# =========================================================
def get_top_list(df, xcol, ycol=None, labelcol="Label", top_n=5):
    sub = df.copy()
    if ycol is not None:
        sub["_score"] = sub[xcol].fillna(0) + sub[ycol].fillna(0)
    else:
        sub["_score"] = sub[xcol].fillna(0)
    sub = sub.sort_values("_score", ascending=False).head(top_n)
    return sub[labelcol].tolist()


def build_summary_values(inventory, structural_df, label_df):
    reg = label_df[label_df["TaskGroup"] == "regression"].copy()
    cls = label_df[label_df["TaskGroup"] == "classification"].copy()

    if not reg.empty:
        reg["Label"] = reg["Dataset"].astype(str) + ":" + reg["Target"].astype(str)
    if not cls.empty:
        cls["Label"] = cls["Dataset"].astype(str) + ":" + cls["Target"].astype(str)

    summary = {
        "n_datasets_total": int(len(inventory)),
        "n_datasets_moleculenet": int((inventory["Benchmark"] == "moleculenet").sum()),
        "n_datasets_admet": int((inventory["Benchmark"] == "admet").sum()),
        "n_datasets_classification": int((inventory["TaskGroup"] == "classification").sum()),
        "n_datasets_regression": int((inventory["TaskGroup"] == "regression").sum()),
        "median_no_pairwise": float(structural_df["No_PairwiseMean"].median()),
        "median_hasmixed_pairwise": float(structural_df["HasMixed_PairwiseMean"].median()),
        "median_true_scaffold_pairwise": float(structural_df["TrueScaffoldClass_PairwiseMean"].median()),
        "median_no_gini": float(structural_df["No_CohesionGini"].median()),
        "median_hasmixed_gini": float(structural_df["HasMixed_CohesionGini"].median()),
        "median_true_scaffold_gini": float(structural_df["TrueScaffoldClass_CohesionGini"].median()),
        "median_same_scaffold_nn": float(structural_df["SameScaffold_NNMean"].median()),
        "median_cross_scaffold_nn": float(structural_df["CrossScaffold_NNMean"].median()),
        "median_no_to_scaffold_nn": float(structural_df["NoToScaffold_NNMean"].median()),
        "n_regression_targets": int(len(reg)),
        "n_classification_targets": int(len(cls)),
        "n_regression_abs_g_ge_0p5": int((reg["Abs_StandardizedMeanDiff"] >= 0.5).sum()) if not reg.empty else 0,
        "n_regression_abs_g_ge_0p8": int((reg["Abs_StandardizedMeanDiff"] >= 0.8).sum()) if not reg.empty else 0,
        "n_classification_prevdiff_ge_0p10": int((cls["Abs_PositiveRate_Diff"] >= 0.10).sum()) if not cls.empty else 0,
        "n_classification_prevdiff_ge_0p20": int((cls["Abs_PositiveRate_Diff"] >= 0.20).sum()) if not cls.empty else 0,
        "top_regression_shift_targets": get_top_list(reg, "Abs_StandardizedMeanDiff", "KS_Statistic", "Label", top_n=5) if not reg.empty else [],
        "top_classification_shift_targets": get_top_list(cls, "Abs_PositiveRate_Diff", "Bernoulli_JSD", "Label", top_n=5) if not cls.empty else [],
    }

    with open(OUT_DIR / "part2_summary_values.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = f"""# Part 2 auto summary (0.08-filtered)

Retained datasets/tasks: **{summary['n_datasets_total']}** total; MoleculeNet **{summary['n_datasets_moleculenet']}**, ADMET **{summary['n_datasets_admet']}**; classification **{summary['n_datasets_classification']}**, regression **{summary['n_datasets_regression']}**.

## Structural-space summary
- No-scaffold median mean pairwise similarity: **{summary['median_no_pairwise']:.3f}**
- Pooled scaffold-bearing median mean pairwise similarity: **{summary['median_hasmixed_pairwise']:.3f}**
- True scaffold-class median mean pairwise similarity: **{summary['median_true_scaffold_pairwise']:.3f}**
- No-scaffold median cohesion Gini: **{summary['median_no_gini']:.3f}**
- Pooled scaffold-bearing median cohesion Gini: **{summary['median_hasmixed_gini']:.3f}**
- True scaffold-class median cohesion Gini: **{summary['median_true_scaffold_gini']:.3f}**
- Median same-scaffold NN similarity: **{summary['median_same_scaffold_nn']:.3f}**
- Median cross-scaffold NN similarity: **{summary['median_cross_scaffold_nn']:.3f}**
- Median no-scaffold-to-scaffold NN similarity: **{summary['median_no_to_scaffold_nn']:.3f}**

## Label-space summary
- Regression targets: **{summary['n_regression_targets']}**
- Regression targets with |Hedges g| >= 0.5: **{summary['n_regression_abs_g_ge_0p5']}**
- Regression targets with |Hedges g| >= 0.8: **{summary['n_regression_abs_g_ge_0p8']}**
- Classification targets: **{summary['n_classification_targets']}**
- Classification targets with |delta positive rate| >= 0.10: **{summary['n_classification_prevdiff_ge_0p10']}**
- Classification targets with |delta positive rate| >= 0.20: **{summary['n_classification_prevdiff_ge_0p20']}**

## Top shifted targets
- Regression, ranked by |Hedges g| + KS: {", ".join(summary['top_regression_shift_targets']) if summary['top_regression_shift_targets'] else "NA"}
- Classification, ranked by |delta positive rate| + JSD: {", ".join(summary['top_classification_shift_targets']) if summary['top_classification_shift_targets'] else "NA"}
"""
    (OUT_DIR / "part2_auto_summary.md").write_text(md, encoding="utf-8")

    return summary


# =========================================================
# 10. Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Compute structural and distribution-shift metrics for no-scaffold molecules.")
    parser.add_argument("--input", type=Path, default=Path("data/processed"), help="Processed dataset directory.")
    parser.add_argument("--out", type=Path, default=Path("results/part2_structural_shift"), help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_paths(args.input, args.out)

    print("Step 1. Collect dataset files...")
    dataset_files = collect_dataset_files()

    print("Step 2. Build dataset inventory and cache...")
    inventory, data_cache = build_dataset_inventory(dataset_files)

    low = inventory[inventory["No_Scaffold_Proportion"] < 0.08]
    if len(low) > 0:
        print("[WARN] Processed input contains datasets below the 0.08 no-scaffold threshold:")
        print(low[["Benchmark", "TaskGroup", "Dataset", "No_Scaffold_Proportion"]].to_string(index=False))

    print("Step 3. Structural metrics...")
    structural_df = build_structural_metrics(inventory, data_cache)

    print("Step 4. Label-shift metrics...")
    label_df = build_label_shift_metrics(inventory, data_cache)

    print("Step 5. Summary values...")
    summary = build_summary_values(inventory, structural_df, label_df)

    print("\nDone.")
    print(f"Outputs saved to: {OUT_DIR}")
    print(f"Retained datasets/tasks: {summary['n_datasets_total']}")
    print(f"Regression targets: {summary['n_regression_targets']}")
    print(f"Classification targets: {summary['n_classification_targets']}")


if __name__ == "__main__":
    main()
