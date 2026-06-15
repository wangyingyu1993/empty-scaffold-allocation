#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ESA configuration sweep.

Input layout:
  data/processed/
    admet/classification/*.csv
    admet/regression/*.csv
    moleculenet/classification/*.csv
    moleculenet/regression/*.csv

  data/splits/datasail/
    <benchmark>/<task_group>/<dataset>/train.csv
    <benchmark>/<task_group>/<dataset>/valid.csv
    <benchmark>/<task_group>/<dataset>/test.csv

Output layout:
  results/esa_condition_sweep/
    esa_condition_sweep_summary.csv
    esa_condition_sweep_metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")

RANDOM_SEED = 20250328
FP_RADIUS = 2
FP_NBITS = 2048
MICROCLUSTER_THRESHOLD = 0.40
SIMILARITY_SAMPLE_CAP = 16


@dataclass(frozen=True)
class ESACondition:
    name: str
    priority: str
    max_cluster_fraction: float
    w_no_cross: float
    w_valid_train: float
    w_test_train: float
    w_size: float
    train_lower: float
    valid_lower: float
    test_lower: float
    drop_budget: float


ESA_SWEEP_CONDITIONS = [
    ESACondition("A_balanced_baseline", "balanced_repair", 0.70, 1.0, 1.0, 1.0, 0.05, 0.80, 0.50, 0.50, 0.05),
    ESACondition("B_test_leakage_priority", "test_leakage_priority", 0.70, 1.0, 1.0, 2.0, 0.05, 0.80, 0.50, 0.50, 0.05),
    ESACondition("C_distribution_uniformity", "distribution_uniformity_priority", 0.60, 1.0, 1.0, 1.0, 0.10, 0.80, 0.60, 0.60, 0.05),
    ESACondition("D_train_support_priority", "train_support_priority", 0.80, 0.5, 1.0, 1.5, 0.03, 0.90, 0.40, 0.40, 0.05),
    ESACondition("E_strict_low_leakage", "strict_low_leakage", 0.50, 1.0, 2.0, 2.0, 0.05, 0.80, 0.50, 0.50, 0.05),
]


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{RANDOM_SEED}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def collect_input_files(input_root: Path, max_datasets: Optional[int] = None) -> pd.DataFrame:
    records = []

    for benchmark in ["admet", "moleculenet"]:
        for task_group in ["classification", "regression"]:
            folder = input_root / benchmark / task_group
            if not folder.exists():
                print(f"[WARN] Missing folder: {folder}")
                continue

            for csv_path in sorted(folder.glob("*.csv")):
                records.append({
                    "Benchmark": benchmark,
                    "TaskGroup": task_group,
                    "Dataset": normalize_name(csv_path.stem),
                    "CSV_Path": csv_path,
                })

    table = pd.DataFrame(records)
    if max_datasets is not None and len(table) > max_datasets:
        table = table.sample(n=max_datasets, random_state=RANDOM_SEED).sort_values(
            ["Benchmark", "TaskGroup", "Dataset"]
        )
    return table.reset_index(drop=True)


def split_weights(n_molecules: int):
    if n_molecules <= 1000:
        return 3, 1, 1
    if n_molecules <= 3000:
        return 5, 1, 1
    return 8, 1, 1


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
        return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS, useChirality=True)
    except Exception:
        return None


def max_similarity_to_set(fp, fps) -> float:
    if fp is None or len(fps) == 0:
        return 0.0
    return float(max(DataStructs.BulkTanimotoSimilarity(fp, fps)))


def mean_query_to_reference_similarity(query_fps, reference_fps) -> float:
    if len(query_fps) == 0 or len(reference_fps) == 0:
        return 0.0
    query = query_fps[:SIMILARITY_SAMPLE_CAP]
    reference = reference_fps[:SIMILARITY_SAMPLE_CAP]
    values = [max_similarity_to_set(fp, reference) for fp in query]
    return float(np.mean(values)) if values else 0.0


def annotate_molecules(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["molecules"] = work["molecules"].astype(str)
    work["_row_id"] = np.arange(len(work))
    work["Scaffold"] = work["molecules"].map(get_scaffold)
    work["Fingerprint"] = work["molecules"].map(make_fingerprint)
    work = work[
        (~work["Scaffold"].isin(["Invalid_SMILES", "Error"]))
        & work["Fingerprint"].notna()
    ].reset_index(drop=True)
    work["ScaffoldStatus"] = np.where(work["Scaffold"] == "No_Scaffold", "No_Scaffold", "Has_Scaffold")
    return work


def read_datasail_split(datasail_root: Path, benchmark: str, task_group: str, dataset: str) -> dict:
    split_dir = datasail_root / benchmark / task_group / dataset
    return {
        branch: pd.read_csv(split_dir / f"{branch}.csv")
        for branch in ["train", "valid", "test"]
    }


def merge_split_with_row_ids(full_annotated: pd.DataFrame, split: dict) -> dict:
    by_smiles = full_annotated.groupby("molecules")["_row_id"].apply(list).to_dict()
    used = set()
    mapped = {}

    for branch in ["train", "valid", "test"]:
        ids = []
        for smiles in split[branch]["molecules"].astype(str):
            candidates = by_smiles.get(smiles, [])
            chosen = None
            for row_id in candidates:
                if row_id not in used:
                    chosen = row_id
                    break
            if chosen is None and candidates:
                chosen = candidates[0]
            if chosen is not None:
                ids.append(chosen)
                used.add(chosen)
        mapped[branch] = set(ids)

    return mapped


def leader_microclusters(no_df: pd.DataFrame, max_cluster_size: int):
    rows = no_df.reset_index(drop=False).rename(columns={"index": "AnnotatedIndex"})
    remaining = list(rows.index)
    clusters = []

    while remaining:
        leader = remaining[0]
        leader_fp = rows.loc[leader, "Fingerprint"]

        members = []
        for idx in remaining:
            sim = DataStructs.TanimotoSimilarity(leader_fp, rows.loc[idx, "Fingerprint"])
            if sim >= MICROCLUSTER_THRESHOLD:
                members.append(idx)

        members = members[:max_cluster_size]
        clusters.append({
            "row_ids": rows.loc[members, "_row_id"].tolist(),
            "fingerprints": rows.loc[members, "Fingerprint"].tolist(),
        })

        member_set = set(members)
        remaining = [idx for idx in remaining if idx not in member_set]

    return clusters


def branch_size_penalty(branch_counts: dict, targets: dict) -> float:
    total = sum(branch_counts.values())
    if total == 0:
        return 0.0
    return float(sum(abs(branch_counts[b] / total - targets[b]) for b in ["train", "valid", "test"]))


def assign_microclusters(clusters, train_backbone_fps, targets, lower_bounds, condition: ESACondition):
    branch_clusters = {"train": [], "valid": [], "test": []}
    branch_counts = {"train": 0, "valid": 0, "test": 0}

    for cluster in clusters:
        best_branch = None
        best_score = None

        for branch in ["train", "valid", "test"]:
            proposed_counts = dict(branch_counts)
            proposed_counts[branch] += len(cluster["row_ids"])

            proposed = {k: list(v) for k, v in branch_clusters.items()}
            proposed[branch] = proposed[branch] + [cluster]

            train_fps = list(train_backbone_fps)
            valid_fps = []
            test_fps = []

            for assigned in proposed["train"]:
                train_fps.extend(assigned["fingerprints"])
            for assigned in proposed["valid"]:
                valid_fps.extend(assigned["fingerprints"])
            for assigned in proposed["test"]:
                test_fps.extend(assigned["fingerprints"])

            no_cross = mean_query_to_reference_similarity(valid_fps, test_fps)
            valid_train = mean_query_to_reference_similarity(valid_fps, train_fps)
            test_train = mean_query_to_reference_similarity(test_fps, train_fps)
            size_penalty = branch_size_penalty(proposed_counts, targets)

            score = (
                condition.w_no_cross * no_cross
                + condition.w_valid_train * valid_train
                + condition.w_test_train * test_train
                + condition.w_size * size_penalty
            )

            if branch_counts[branch] < lower_bounds[branch]:
                score -= 1.0

            if best_score is None or score < best_score:
                best_score = score
                best_branch = branch

        branch_clusters[best_branch].append(cluster)
        branch_counts[best_branch] += len(cluster["row_ids"])

    return branch_clusters


def evaluate_condition(full_df: pd.DataFrame, raw_split: dict, condition: ESACondition):
    annotated = annotate_molecules(full_df)
    split_row_ids = merge_split_with_row_ids(annotated, raw_split)

    no_df = annotated[annotated["ScaffoldStatus"] == "No_Scaffold"].copy()
    has_df = annotated[annotated["ScaffoldStatus"] == "Has_Scaffold"].copy()

    if no_df.empty:
        return {
            "condition": condition.name,
            "priority": condition.priority,
            "n_no_scaffold": 0,
            "support_recovery": np.nan,
            "sink_ratio": np.nan,
            "max_eval_train_similarity": np.nan,
            "score": np.nan,
        }

    backbone = {}
    for branch in ["train", "valid", "test"]:
        backbone[branch] = has_df[has_df["_row_id"].isin(split_row_ids[branch])].copy()

    weights = split_weights(len(annotated))
    total_weight = sum(weights)
    targets = {
        "train": weights[0] / total_weight,
        "valid": weights[1] / total_weight,
        "test": weights[2] / total_weight,
    }

    target_no_counts = {
        branch: max(1, int(math.floor(len(no_df) * targets[branch])))
        for branch in ["train", "valid", "test"]
    }
    lower_bounds = {
        "train": int(math.floor(target_no_counts["train"] * condition.train_lower)),
        "valid": int(math.floor(target_no_counts["valid"] * condition.valid_lower)),
        "test": int(math.floor(target_no_counts["test"] * condition.test_lower)),
    }

    smallest_target = max(1, min(target_no_counts.values()))
    max_cluster_size = max(1, int(math.ceil(condition.max_cluster_fraction * smallest_target)))

    clusters = leader_microclusters(no_df, max_cluster_size=max_cluster_size)
    clusters = sorted(clusters, key=lambda x: len(x["row_ids"]), reverse=True)

    train_backbone_fps = backbone["train"]["Fingerprint"].tolist()
    branch_clusters = assign_microclusters(clusters, train_backbone_fps, targets, lower_bounds, condition)

    branch_row_ids = {}
    for branch in ["train", "valid", "test"]:
        has_ids = backbone[branch]["_row_id"].tolist()
        no_ids = [row_id for cluster in branch_clusters[branch] for row_id in cluster["row_ids"]]
        branch_row_ids[branch] = sorted(has_ids + no_ids)

    full_no_prop = float((annotated["ScaffoldStatus"] == "No_Scaffold").mean())

    branch_props = {}
    for branch in ["train", "valid", "test"]:
        branch_annotated = annotated[annotated["_row_id"].isin(branch_row_ids[branch])]
        branch_props[branch] = float((branch_annotated["ScaffoldStatus"] == "No_Scaffold").mean()) if len(branch_annotated) else np.nan

    train_fps = annotated[annotated["_row_id"].isin(branch_row_ids["train"])]["Fingerprint"].tolist()
    valid_no_fps = annotated[
        annotated["_row_id"].isin(branch_row_ids["valid"])
        & (annotated["ScaffoldStatus"] == "No_Scaffold")
    ]["Fingerprint"].tolist()
    test_no_fps = annotated[
        annotated["_row_id"].isin(branch_row_ids["test"])
        & (annotated["ScaffoldStatus"] == "No_Scaffold")
    ]["Fingerprint"].tolist()

    valid_train_similarity = mean_query_to_reference_similarity(valid_no_fps, train_fps)
    test_train_similarity = mean_query_to_reference_similarity(test_no_fps, train_fps)
    max_eval_train_similarity = max(valid_train_similarity, test_train_similarity)

    support_recovery = branch_props["train"] / full_no_prop if full_no_prop else np.nan
    sink_ratio = max(branch_props["valid"], branch_props["test"]) / branch_props["train"] if branch_props["train"] else np.nan

    distribution_gap = max(abs(branch_props[b] - full_no_prop) for b in ["train", "valid", "test"])
    score = (
        abs(support_recovery - 1.0)
        + sink_ratio
        + distribution_gap
        + max_eval_train_similarity
    )

    return {
        "condition": condition.name,
        "priority": condition.priority,
        "n_no_scaffold": int(len(no_df)),
        "n_microclusters": int(len(clusters)),
        "full_no_scaffold_proportion": full_no_prop,
        "train_no_scaffold_proportion": branch_props["train"],
        "valid_no_scaffold_proportion": branch_props["valid"],
        "test_no_scaffold_proportion": branch_props["test"],
        "support_recovery": support_recovery,
        "sink_ratio": sink_ratio,
        "valid_train_similarity": valid_train_similarity,
        "test_train_similarity": test_train_similarity,
        "max_eval_train_similarity": max_eval_train_similarity,
        "distribution_gap": distribution_gap,
        "score": score,
    }


def process_dataset(csv_path: Path, datasail_root: Path, benchmark: str, task_group: str, dataset: str):
    full_df = pd.read_csv(csv_path)
    if "molecules" not in full_df.columns:
        raise ValueError(f"{csv_path} does not contain a molecules column")

    raw_split = read_datasail_split(datasail_root, benchmark, task_group, dataset)

    rows = []
    for condition in ESA_SWEEP_CONDITIONS:
        metrics = evaluate_condition(full_df, raw_split, condition)
        rows.append({
            "Benchmark": benchmark,
            "TaskGroup": task_group,
            "Dataset": dataset,
            **metrics,
        })

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run ESA configuration sweep.")
    parser.add_argument("--input-root", type=Path, default=Path("data/processed"), help="Processed dataset directory.")
    parser.add_argument("--datasail-root", type=Path, default=Path("data/splits/datasail"), help="DataSAIL split directory.")
    parser.add_argument("--out", type=Path, default=Path("results/esa_condition_sweep"), help="Output directory.")
    parser.add_argument("--max-datasets", type=int, default=None, help="Optional number of datasets for a fixed-size sweep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dataset_files = collect_input_files(args.input_root, max_datasets=args.max_datasets)
    if dataset_files.empty:
        raise SystemExit(f"No dataset CSV files found under {args.input_root}")

    rows = []

    for _, item in dataset_files.iterrows():
        rows.extend(process_dataset(
            csv_path=item["CSV_Path"],
            datasail_root=args.datasail_root,
            benchmark=item["Benchmark"],
            task_group=item["TaskGroup"],
            dataset=item["Dataset"],
        ))
        print(f"Evaluated ESA conditions: {item['Benchmark']}/{item['TaskGroup']}/{item['Dataset']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "esa_condition_sweep_summary.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "input_root": str(args.input_root),
        "datasail_root": str(args.datasail_root),
        "output_directory": str(args.out),
        "random_seed": RANDOM_SEED,
        "max_datasets": args.max_datasets,
        "conditions": [asdict(condition) for condition in ESA_SWEEP_CONDITIONS],
    }
    (args.out / "esa_condition_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Evaluated datasets: {len(dataset_files)}")
    print(f"Output directory: {args.out}")


if __name__ == "__main__":
    main()
