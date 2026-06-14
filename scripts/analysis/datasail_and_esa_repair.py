#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate DataSAIL+ESA repaired splits for full datasets.

The script keeps scaffold-bearing molecules in a DataSAIL-derived backbone
(DataSAIL + SCIP when available, otherwise a scaffold-group fallback), partitions
no-scaffold molecules into local micro-clusters, and assigns those micro-clusters
to train/valid/test/drop with the coupled greedy ESA procedure. Each
(condition, dataset) task is executed in an isolated subprocess so one failure
does not stop the batch.

Outputs
-------
<output-root>/<condition>/<relative-input-path>/<dataset>/train.csv
<output-root>/<condition>/<relative-input-path>/<dataset>/valid.csv
<output-root>/<condition>/<relative-input-path>/<dataset>/test.csv
<output-root>/<condition>/<relative-input-path>/<dataset>/split_assignment_audit.csv
<output-root>/<condition>/<relative-input-path>/<dataset>/esa_coupled_cluster_summary.csv
<output-root>/combined_condition_summary.csv
<output-root>/failed_tasks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters.rdSimDivPickers import LeaderPicker

try:
    from datasail.sail import datasail
except Exception:
    datasail = None

rdBase.BlockLogs()

# =========================
# Parameters / constants
# =========================
SMILES_COLUMN = "molecules"
SPLIT_NAMES = ["train", "valid", "test"]
NO_SCAFFOLD_LABELS = {"Invalid_SMILES", "No_Scaffold", "Error"}

# Fixed ESA defaults (single-threshold 0.40)
ESA_FP_RADIUS = 2
ESA_FP_BITS = 2048
ESA_INIT_SIM_THRESHOLD = 0.40
ESA_RECLUSTER_STEP = 0.00
ESA_MAX_SIM_THRESHOLD = 0.40
ESA_MAX_REFINE_DEPTH = 1

# DataSAIL defaults for scaffold-bearing split
DATASAIL_TECHNIQUES = ["C1e"]
DATASAIL_SOLVER = "SCIP"
DATASAIL_E_TYPE = "M"
DATASAIL_RUNS = 1

# Two selected conditions
CONDITIONS: Dict[str, Dict[str, float]] = {
    "A_balanced_baseline": {
        "esa_max_cluster_to_smallest_target": 0.70,
        "lambda_internal": 1.0,
        "lambda_valid_trainall": 1.0,
        "lambda_test_trainall": 1.0,
        "lambda_imbalance": 0.05,
        "lambda_drop": 2.0,
        "lower_train_frac": 0.80,
        "lower_valid_frac": 0.50,
        "lower_test_frac": 0.50,
        "drop_budget_frac": 0.05,
        "sample_cap": 16,
    },
    "D_train_support_priority": {
        "esa_max_cluster_to_smallest_target": 0.80,
        "lambda_internal": 0.5,
        "lambda_valid_trainall": 1.0,
        "lambda_test_trainall": 1.5,
        "lambda_imbalance": 0.03,
        "lambda_drop": 2.0,
        "lower_train_frac": 0.90,
        "lower_valid_frac": 0.40,
        "lower_test_frac": 0.40,
        "drop_budget_frac": 0.05,
        "sample_cap": 16,
    },
}


# =========================
# Generic helpers
# =========================
def mol2smiles(mol):
    try:
        return Chem.MolToSmiles(Chem.rdmolops.RemoveHs(mol))
    except Exception:
        return None


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid_SMILES"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "No_Scaffold"
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return "Error"


def smiles_to_fp(smiles: str, radius: int = ESA_FP_RADIUS, n_bits: int = ESA_FP_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    except Exception:
        return None


def get_split_ratio_and_weights(n: int) -> Tuple[Tuple[float, float, float], List[int]]:
    if n <= 1000:
        return (0.6, 0.2, 0.2), [3, 1, 1]
    elif n <= 3000:
        return (5 / 7, 1 / 7, 1 / 7), [5, 1, 1]
    else:
        return (0.8, 0.1, 0.1), [8, 1, 1]


def counts_from_ratios(n: int, ratios: Tuple[float, float, float]) -> Dict[str, int]:
    raw = np.array(ratios, dtype=float) * n
    base = np.floor(raw).astype(int)
    remain = int(n - base.sum())
    frac = raw - base
    order = np.argsort(-frac)
    for idx in order[:remain]:
        base[idx] += 1
    return {split: int(cnt) for split, cnt in zip(SPLIT_NAMES, base.tolist())}


def summarize_split_df(df_part: pd.DataFrame):
    total_count = len(df_part)
    no_scaffold_count = int(df_part["Scaffold"].isin(NO_SCAFFOLD_LABELS).sum())
    scaffold_bearing_count = int(total_count - no_scaffold_count)
    proportion = (no_scaffold_count / total_count) if total_count > 0 else 0.0
    return total_count, scaffold_bearing_count, no_scaffold_count, proportion


def find_all_csv_files(root: Path):
    return sorted(root.rglob("*.csv"))


def build_output_dir(input_root: Path, output_root: Path, csv_path: Path):
    rel_path = csv_path.relative_to(input_root)
    rel_parent = rel_path.parent
    dataset_name = csv_path.stem
    save_dir = output_root / rel_parent / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def summary_json_path(input_root: Path, output_root: Path, csv_path: Path) -> Path:
    return build_output_dir(input_root, output_root, csv_path) / "summary.json"


def safe_tail(text: str, max_chars: int = 4000) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]




def write_dict_rows_csv(rows: List[Dict], path: Path) -> None:
    """Write a list of dict rows to CSV using the standard library only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_failed_rows(rows: List[Dict]) -> None:
    if not rows:
        print("\n[OK] No failed tasks.")
        return
    print("\n[FAILED TASKS]")
    cols = ["condition", "dataset", "returncode", "error_type", "log_path"]
    widths = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))

# =========================
# Scaffold-bearing split via DataSAIL / fallback
# =========================
def extract_datasail_assignment(e_splits) -> Dict[str, str]:
    if isinstance(e_splits, dict):
        first_val = next(iter(e_splits.values()))
        if isinstance(first_val, list) and len(first_val) > 0 and isinstance(first_val[0], dict):
            return first_val[0]
        if isinstance(first_val, dict):
            return first_val
    raise ValueError(f"Cannot parse DataSAIL return value: {type(e_splits)} -> {e_splits}")


def choose_best_split_for_group(group_size: int, actual_counts: Dict[str, int], target_counts: Dict[str, int]) -> str:
    best_split = None
    best_score = None
    for split in SPLIT_NAMES:
        target = max(target_counts[split], 1)
        now = actual_counts[split]
        after = now + group_size
        overflow = max(0, after - target_counts[split]) / target
        deviation = abs(after - target_counts[split]) / target
        score = 3.0 * overflow + deviation
        if best_score is None or score < best_score - 1e-12:
            best_split, best_score = split, score
        elif abs(score - best_score) <= 1e-12 and actual_counts[split] < actual_counts[best_split]:
            best_split, best_score = split, score
    return best_split


def greedy_assign_groups_to_splits(group_to_ids: Dict[str, List[str]], target_counts: Dict[str, int]) -> Dict[str, str]:
    actual_counts = {s: 0 for s in SPLIT_NAMES}
    id_to_split = {}
    ordered_groups = sorted(group_to_ids.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for _, ids in ordered_groups:
        split = choose_best_split_for_group(len(ids), actual_counts, target_counts)
        for _id in ids:
            id_to_split[_id] = split
        actual_counts[split] += len(ids)
    return id_to_split


def fallback_group_split_by_scaffold(df_scaffold: pd.DataFrame, ratios: Tuple[float, float, float]) -> Dict[str, str]:
    target_counts = counts_from_ratios(len(df_scaffold), ratios)
    group_to_ids = df_scaffold.groupby("Scaffold")["ID"].apply(list).to_dict()
    return greedy_assign_groups_to_splits(group_to_ids, target_counts)


def datasail_split_scaffold_bearing(df_scaffold: pd.DataFrame, weights: List[int], ratios: Tuple[float, float, float]) -> Tuple[Dict[str, str], str]:
    if len(df_scaffold) == 0:
        return {}, "no_scaffold_only"
    if datasail is None:
        return fallback_group_split_by_scaffold(df_scaffold, ratios), "fallback_scaffold_group_no_datasail"

    df_sail = df_scaffold[["ID", "SMILES_norm"]].copy()
    e_data = dict(df_sail.values.tolist())
    try:
        e_splits, _, _ = datasail(
            techniques=DATASAIL_TECHNIQUES,
            splits=weights,
            names=SPLIT_NAMES,
            runs=DATASAIL_RUNS,
            solver=DATASAIL_SOLVER,
            e_type=DATASAIL_E_TYPE,
            e_data=e_data,
        )
        assignment = extract_datasail_assignment(e_splits)
        return assignment, "datasail"
    except Exception as e:
        print(f"[WARN] DataSAIL failed on scaffold-bearing subset, fallback to scaffold-group greedy split: {e}")
        return fallback_group_split_by_scaffold(df_scaffold, ratios), "fallback_scaffold_group"


# =========================
# ESA micro-clusters for no-scaffold molecules
# =========================
def pick_leaders(fps: List, sim_threshold: float) -> List[int]:
    if len(fps) == 0:
        return []
    if len(fps) == 1:
        return [0]
    picker = LeaderPicker()
    distance_threshold = max(0.0, min(1.0, 1.0 - sim_threshold))
    leaders = list(picker.LazyBitVectorPick(fps, len(fps), distance_threshold))
    return leaders or [0]


def assign_members_to_leaders(fps: List, leader_indices: List[int]) -> List[List[int]]:
    if len(fps) == 0:
        return []
    if len(leader_indices) == 0:
        return [list(range(len(fps)))]
    leader_fps = [fps[i] for i in leader_indices]
    clusters = {leader_idx: [] for leader_idx in leader_indices}
    for i, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, leader_fps)
        best_pos = int(np.argmax(sims))
        best_leader = leader_indices[best_pos]
        clusters[best_leader].append(i)
    return list(clusters.values())


def similarity_sorted_chunks(global_indices: List[int], all_fps: List, chunk_size: int) -> List[List[int]]:
    if len(global_indices) <= chunk_size:
        return [global_indices]
    rep_idx = global_indices[0]
    rep_fp = all_fps[rep_idx]
    sims = []
    for idx in global_indices:
        sim = DataStructs.TanimotoSimilarity(all_fps[idx], rep_fp)
        sims.append((idx, sim))
    ordered = [idx for idx, _ in sorted(sims, key=lambda x: (-x[1], x[0]))]
    return [ordered[i:i + chunk_size] for i in range(0, len(ordered), chunk_size)]


def refine_large_cluster(
    global_indices: List[int],
    all_fps: List,
    sim_threshold: float,
    max_cluster_size: int,
    depth: int = 0,
) -> List[List[int]]:
    if len(global_indices) <= max_cluster_size or len(global_indices) <= 1:
        return [global_indices]
    if depth >= ESA_MAX_REFINE_DEPTH:
        return similarity_sorted_chunks(global_indices, all_fps, max_cluster_size)

    tighter_sim = min(ESA_MAX_SIM_THRESHOLD, sim_threshold + ESA_RECLUSTER_STEP)
    local_fps = [all_fps[i] for i in global_indices]
    leader_local_idx = pick_leaders(local_fps, tighter_sim)
    if len(leader_local_idx) <= 1:
        return similarity_sorted_chunks(global_indices, all_fps, max_cluster_size)

    local_clusters = assign_members_to_leaders(local_fps, leader_local_idx)
    if len(local_clusters) <= 1:
        return similarity_sorted_chunks(global_indices, all_fps, max_cluster_size)

    refined = []
    for local_cluster in local_clusters:
        sub_global_indices = [global_indices[i] for i in local_cluster]
        refined.extend(refine_large_cluster(sub_global_indices, all_fps, tighter_sim, max_cluster_size, depth + 1))
    return refined


def choose_max_cluster_size(target_counts: Dict[str, int], esa_max_cluster_to_smallest_target: float) -> int:
    nonzero_targets = [v for v in target_counts.values() if v > 0]
    if not nonzero_targets:
        return 1
    smallest_target = min(nonzero_targets)
    return max(1, int(math.floor(smallest_target * esa_max_cluster_to_smallest_target)))


def build_esa_microclusters(fps: List, target_counts: Dict[str, int], esa_max_cluster_to_smallest_target: float) -> List[List[int]]:
    if len(fps) == 0:
        return []
    if len(fps) == 1:
        return [[0]]

    max_cluster_size = choose_max_cluster_size(target_counts, esa_max_cluster_to_smallest_target)
    leaders = pick_leaders(fps, ESA_INIT_SIM_THRESHOLD)
    coarse_clusters = assign_members_to_leaders(fps, leaders)

    refined_clusters = []
    for cluster in coarse_clusters:
        refined_clusters.extend(refine_large_cluster(cluster, fps, ESA_INIT_SIM_THRESHOLD, max_cluster_size, 0))

    refined_clusters = sorted(refined_clusters, key=lambda x: (-len(x), x[0] if len(x) > 0 else -1))
    return refined_clusters


# =========================
# Similarity estimation helpers for coupled greedy assignment
# =========================
def sample_indices(indices: List[int], cap: int, rng: np.random.Generator) -> List[int]:
    if len(indices) <= cap:
        return list(indices)
    return sorted(rng.choice(indices, size=cap, replace=False).tolist())


def estimate_group_to_train_similarity(group_indices: List[int], all_fps: List, train_fps: List, sample_cap: int, rng: np.random.Generator) -> float:
    if len(group_indices) == 0 or len(train_fps) == 0:
        return 0.0
    idx_g = sample_indices(group_indices, sample_cap, rng)
    sampled_train = train_fps if len(train_fps) <= sample_cap else [train_fps[i] for i in sorted(rng.choice(len(train_fps), size=sample_cap, replace=False).tolist())]
    vals = []
    for i in idx_g:
        vals.extend(DataStructs.BulkTanimotoSimilarity(all_fps[i], sampled_train))
    if not vals:
        return 0.0
    return float(np.mean(vals)) * len(group_indices) * max(1, len(sampled_train))


def estimate_group_pair_similarity(group_a: List[int], group_b: List[int], all_fps: List, sample_cap: int, rng: np.random.Generator) -> float:
    if len(group_a) == 0 or len(group_b) == 0:
        return 0.0
    idx_a = sample_indices(group_a, sample_cap, rng)
    idx_b = sample_indices(group_b, sample_cap, rng)
    fps_b = [all_fps[i] for i in idx_b]
    vals = []
    for i in idx_a:
        vals.extend(DataStructs.BulkTanimotoSimilarity(all_fps[i], fps_b))
    if not vals:
        return 0.0
    return float(np.mean(vals)) * len(group_a) * len(group_b)


# =========================
# Coupled no-scaffold assignment (greedy version)
# =========================
def coupled_assign_no_scaffold(
    df_no_scaffold: pd.DataFrame,
    ratios: Tuple[float, float, float],
    scaffold_bearing_train_fps: List,
    *,
    seed: int,
    sample_cap: int,
    lambda_internal: float,
    lambda_valid_trainall: float,
    lambda_test_trainall: float,
    lambda_imbalance: float,
    lambda_drop: float,
    lower_train_frac: float,
    lower_valid_frac: float,
    lower_test_frac: float,
    drop_budget_frac: float,
    esa_max_cluster_to_smallest_target: float,
) -> Tuple[Dict[str, str], Dict[str, str], pd.DataFrame]:
    if len(df_no_scaffold) == 0:
        empty = pd.DataFrame(columns=["cluster_id", "split", "cluster_size", "const_to_sb_train", "is_invalid_fp_group"])
        return {}, {}, empty

    rng = np.random.default_rng(seed)
    target_counts = counts_from_ratios(len(df_no_scaffold), ratios)
    lower_bounds = {
        "train": max(1, int(math.floor(target_counts["train"] * lower_train_frac))) if target_counts["train"] > 0 else 0,
        "valid": max(1, int(math.floor(target_counts["valid"] * lower_valid_frac))) if target_counts["valid"] > 0 else 0,
        "test": max(1, int(math.floor(target_counts["test"] * lower_test_frac))) if target_counts["test"] > 0 else 0,
    }
    drop_budget = int(math.floor(len(df_no_scaffold) * drop_budget_frac))

    ids = df_no_scaffold["ID"].tolist()
    smiles_list = df_no_scaffold["SMILES_norm"].tolist()

    fps_all = []
    valid_local_indices = []
    invalid_local_indices = []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_fp(smi)
        if fp is None:
            invalid_local_indices.append(i)
        else:
            valid_local_indices.append(i)
            fps_all.append(fp)

    groups: List[Dict] = []
    id_to_cluster_id: Dict[str, str] = {}

    if len(valid_local_indices) > 0:
        local_clusters = build_esa_microclusters(fps_all, target_counts, esa_max_cluster_to_smallest_target)
        for cid, cluster_local in enumerate(local_clusters, start=1):
            cluster_name = f"ESA_{cid:05d}"
            member_ids = [ids[valid_local_indices[j]] for j in cluster_local]
            for _id in member_ids:
                id_to_cluster_id[_id] = cluster_name
            groups.append({
                "cluster_id": cluster_name,
                "member_ids": member_ids,
                "member_fp_local_idx": list(cluster_local),
                "size": len(member_ids),
                "is_invalid_fp_group": False,
            })

    start_idx = len(groups) + 1
    for offset, local_idx in enumerate(invalid_local_indices):
        cluster_name = f"ESA_{start_idx + offset:05d}"
        _id = ids[local_idx]
        id_to_cluster_id[_id] = cluster_name
        groups.append({
            "cluster_id": cluster_name,
            "member_ids": [_id],
            "member_fp_local_idx": [],
            "size": 1,
            "is_invalid_fp_group": True,
        })

    if not groups:
        empty = pd.DataFrame(columns=["cluster_id", "split", "cluster_size", "const_to_sb_train", "is_invalid_fp_group"])
        return {}, {}, empty

    const_to_sb_train = []
    for g in groups:
        if g["is_invalid_fp_group"] or len(scaffold_bearing_train_fps) == 0:
            const_to_sb_train.append(0.0)
        else:
            const_to_sb_train.append(
                estimate_group_to_train_similarity(g["member_fp_local_idx"], fps_all, scaffold_bearing_train_fps, sample_cap, rng)
            )

    pair_sim = np.zeros((len(groups), len(groups)), dtype=float)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i]["is_invalid_fp_group"] or groups[j]["is_invalid_fp_group"]:
                sim = 0.0
            else:
                sim = estimate_group_pair_similarity(groups[i]["member_fp_local_idx"], groups[j]["member_fp_local_idx"], fps_all, sample_cap, rng)
            pair_sim[i, j] = pair_sim[j, i] = sim

    order = sorted(range(len(groups)), key=lambda i: (-groups[i]["size"], -const_to_sb_train[i], groups[i]["cluster_id"]))

    assignments: Dict[int, str] = {}
    split_counts = {s: 0 for s in SPLIT_NAMES}
    dropped_count = 0
    assigned_by_split = {s: [] for s in SPLIT_NAMES}

    def remaining_size_from(pos: int) -> int:
        return int(sum(groups[idx]["size"] for idx in order[pos:]))

    def feasible_after(candidate_split: str, gidx: int, pos: int) -> bool:
        cand_counts = split_counts.copy()
        cand_dropped = dropped_count
        cand_size = groups[gidx]["size"]
        if candidate_split == "drop":
            cand_dropped += cand_size
            if cand_dropped > drop_budget:
                return False
        else:
            cand_counts[candidate_split] += cand_size

        remaining = remaining_size_from(pos + 1)
        deficit = sum(max(0, lower_bounds[s] - cand_counts[s]) for s in SPLIT_NAMES)
        if deficit > remaining:
            return False
        if cand_dropped > drop_budget:
            return False
        return True

    def imbalance_cost(after_counts: Dict[str, int]) -> float:
        cost = 0.0
        for s in SPLIT_NAMES:
            denom = max(target_counts[s], 1)
            overflow = max(0, after_counts[s] - target_counts[s]) / denom
            deviation = abs(after_counts[s] - target_counts[s]) / denom
            cost += (3.0 * overflow + deviation)
        return lambda_imbalance * cost

    def pairwise_increment(candidate_split: str, gidx: int) -> float:
        cost = 0.0
        if candidate_split == "drop":
            return lambda_drop * groups[gidx]["size"]
        if candidate_split == "valid":
            cost += lambda_valid_trainall * const_to_sb_train[gidx]
        elif candidate_split == "test":
            cost += lambda_test_trainall * const_to_sb_train[gidx]

        for s in SPLIT_NAMES:
            for j in assigned_by_split[s]:
                sim = pair_sim[gidx, j]
                if sim <= 0:
                    continue
                if {candidate_split, s} == {"train", "valid"}:
                    cost += (lambda_internal + lambda_valid_trainall) * sim
                elif {candidate_split, s} == {"train", "test"}:
                    cost += (lambda_internal + lambda_test_trainall) * sim
                elif {candidate_split, s} == {"valid", "test"}:
                    cost += lambda_internal * sim
        return cost

    for pos, gidx in enumerate(order):
        gsize = groups[gidx]["size"]
        best_split = None
        best_score = None
        for cand in ["train", "valid", "test", "drop"]:
            if not feasible_after(cand, gidx, pos):
                continue

            after_counts = split_counts.copy()
            if cand != "drop":
                after_counts[cand] += gsize

            deficit_after = sum(max(0, lower_bounds[s] - after_counts[s]) for s in SPLIT_NAMES)
            deficit_penalty = deficit_after / max(1, len(df_no_scaffold))
            unmet_splits = sum(1 for s in SPLIT_NAMES if after_counts[s] < lower_bounds[s])
            drop_extra = 10.0 * unmet_splits if cand == "drop" and unmet_splits > 0 else 0.0

            score = pairwise_increment(cand, gidx) + imbalance_cost(after_counts) + 2.0 * deficit_penalty + drop_extra

            if best_score is None or score < best_score - 1e-12:
                best_split, best_score = cand, score
            elif abs(score - best_score) <= 1e-12:
                if best_split == "drop" and cand != "drop":
                    best_split, best_score = cand, score
                elif cand != "drop" and best_split != "drop":
                    if split_counts[cand] < split_counts[best_split]:
                        best_split, best_score = cand, score

        if best_split is None:
            ranked = ["train", "valid", "test", "drop"]
            best_split = ranked[0]
            best_score = float("inf")
            for cand in ranked:
                after_counts = split_counts.copy()
                if cand != "drop":
                    after_counts[cand] += gsize
                score = pairwise_increment(cand, gidx) + imbalance_cost(after_counts)
                if score < best_score:
                    best_split, best_score = cand, score

        assignments[gidx] = best_split
        if best_split == "drop":
            dropped_count += gsize
        else:
            split_counts[best_split] += gsize
            assigned_by_split[best_split].append(gidx)

    needy = [s for s in SPLIT_NAMES if split_counts[s] < lower_bounds[s]]
    if needy:
        train_groups = list(assigned_by_split["train"])
        train_groups_sorted = sorted(train_groups, key=lambda i: (groups[i]["size"], const_to_sb_train[i]))
        for need_split in needy:
            while split_counts[need_split] < lower_bounds[need_split] and train_groups_sorted:
                gidx = train_groups_sorted.pop(0)
                gsize = groups[gidx]["size"]
                if split_counts["train"] - gsize < lower_bounds["train"]:
                    continue
                assignments[gidx] = need_split
                split_counts["train"] -= gsize
                split_counts[need_split] += gsize
                if gidx in assigned_by_split["train"]:
                    assigned_by_split["train"].remove(gidx)
                assigned_by_split[need_split].append(gidx)

    id_to_split: Dict[str, str] = {}
    cluster_rows = []
    for i, g in enumerate(groups):
        split = assignments.get(i, "drop")
        for _id in g["member_ids"]:
            id_to_split[_id] = split
        cluster_rows.append({
            "cluster_id": g["cluster_id"],
            "split": split,
            "cluster_size": g["size"],
            "const_to_sb_train": float(const_to_sb_train[i]),
            "is_invalid_fp_group": bool(g["is_invalid_fp_group"]),
        })

    cluster_summary = pd.DataFrame(cluster_rows)
    return id_to_split, id_to_cluster_id, cluster_summary


# =========================
# Dataset processing
# =========================
def process_dataset(
    csv_path: Path,
    input_root: Path,
    output_root: Path,
    *,
    condition_name: str,
    condition_params: Dict[str, float],
    smiles_column: str,
    seed: int,
):
    dataset_name = csv_path.stem
    print(f"\n{'=' * 100}\n[CONDITION] {condition_name}\n[DATASET] {csv_path}")

    df_raw = pd.read_csv(csv_path)
    if len(df_raw) > 60000:
        df_raw = df_raw.sample(n=50000, random_state=42).reset_index(drop=True)
    if smiles_column not in df_raw.columns:
        print(f"[SKIP] Missing column '{smiles_column}'.")
        return None

    original_columns = df_raw.columns.tolist()
    df = df_raw.dropna(subset=[smiles_column]).copy().reset_index(drop=True)
    df["SMILES_norm"] = df[smiles_column].apply(lambda x: mol2smiles(Chem.MolFromSmiles(str(x))))
    df = df.dropna(subset=["SMILES_norm"]).reset_index(drop=True)
    if len(df) == 0:
        print("[SKIP] No valid molecules after cleaning.")
        return None

    n = len(df)
    ratios, weights = get_split_ratio_and_weights(n)
    target_total_counts = counts_from_ratios(n, ratios)

    df["Scaffold"] = df["SMILES_norm"].astype(str).apply(get_scaffold)
    df["Is_NoScaffold"] = df["Scaffold"].isin(NO_SCAFFOLD_LABELS)
    df["ID"] = [f"{dataset_name}_{i + 1:06d}" for i in range(len(df))]

    df_scaffold = df[~df["Is_NoScaffold"]].copy().reset_index(drop=True)
    df_no_scaffold = df[df["Is_NoScaffold"]].copy().reset_index(drop=True)

    print(f"[INFO] n_total={n}, scaffold-bearing={len(df_scaffold)}, no-scaffold={len(df_no_scaffold)}")
    print(f"[INFO] ratios={ratios}, weights={weights}, target_total_counts={target_total_counts}")

    scaffold_id_to_split, scaffold_method = datasail_split_scaffold_bearing(df_scaffold, weights, ratios)
    print(f"[INFO] scaffold split method = {scaffold_method}")

    sb_train_ids = {k for k, v in scaffold_id_to_split.items() if v == "train"}
    sb_train_fps = []
    if len(df_scaffold) > 0:
        for _, row in df_scaffold.iterrows():
            if row["ID"] in sb_train_ids:
                fp = smiles_to_fp(row["SMILES_norm"])
                if fp is not None:
                    sb_train_fps.append(fp)

    ns_id_to_split, ns_id_to_cluster, ns_cluster_summary = coupled_assign_no_scaffold(
        df_no_scaffold=df_no_scaffold,
        ratios=ratios,
        scaffold_bearing_train_fps=sb_train_fps,
        seed=seed,
        sample_cap=int(condition_params["sample_cap"]),
        lambda_internal=float(condition_params["lambda_internal"]),
        lambda_valid_trainall=float(condition_params["lambda_valid_trainall"]),
        lambda_test_trainall=float(condition_params["lambda_test_trainall"]),
        lambda_imbalance=float(condition_params["lambda_imbalance"]),
        lambda_drop=float(condition_params["lambda_drop"]),
        lower_train_frac=float(condition_params["lower_train_frac"]),
        lower_valid_frac=float(condition_params["lower_valid_frac"]),
        lower_test_frac=float(condition_params["lower_test_frac"]),
        drop_budget_frac=float(condition_params["drop_budget_frac"]),
        esa_max_cluster_to_smallest_target=float(condition_params["esa_max_cluster_to_smallest_target"]),
    )

    all_id_to_split = {}
    all_id_to_split.update(scaffold_id_to_split)
    all_id_to_split.update(ns_id_to_split)

    df["Split"] = df["ID"].map(all_id_to_split)
    df["Split_source"] = np.where(df["Is_NoScaffold"], f"ESA_{condition_name}", "DataSAIL_scaffold")
    df["ESA_cluster_id"] = df["ID"].map(ns_id_to_cluster)

    save_dir = build_output_dir(input_root, output_root, csv_path)

    split_stats = {}
    for split_name in SPLIT_NAMES:
        split_df = df[df["Split"] == split_name].copy()
        total_count, scaffold_count, no_scaffold_count, proportion = summarize_split_df(split_df)
        split_stats[split_name] = {
            "count": total_count,
            "scaffold_bearing_count": scaffold_count,
            "no_scaffold_count": no_scaffold_count,
            "no_scaffold_proportion": proportion,
        }
        split_df[original_columns].to_csv(save_dir / f"{split_name}.csv", index=False)
        print(
            f"[SPLIT] {split_name}: total={total_count}, scaffold-bearing={scaffold_count}, "
            f"no-scaffold={no_scaffold_count}, ratio={proportion:.2%}"
        )

    dropped_df = df[df["Split"] == "drop"].copy()
    if len(dropped_df) > 0:
        dropped_df[original_columns].to_csv(save_dir / "dropped.csv", index=False)
        print(f"[DROP] dropped={len(dropped_df)}")

    audit_cols = [
        "ID", smiles_column, "SMILES_norm", "Scaffold", "Is_NoScaffold",
        "Split", "Split_source", "ESA_cluster_id"
    ]
    df[audit_cols].to_csv(save_dir / "split_assignment_audit.csv", index=False, encoding="utf-8-sig")
    ns_cluster_summary.to_csv(save_dir / "esa_coupled_cluster_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "condition": condition_name,
        "dataset": dataset_name,
        "n_total": int(n),
        "n_scaffold_bearing": int(len(df_scaffold)),
        "n_no_scaffold": int(len(df_no_scaffold)),
        "ratios": {k: float(v) for k, v in zip(SPLIT_NAMES, ratios)},
        "weights": {k: int(v) for k, v in zip(SPLIT_NAMES, weights)},
        "target_total_counts": target_total_counts,
        "scaffold_split_method": scaffold_method,
        "condition_params": condition_params,
        "actual": {s: int(split_stats[s]["count"]) for s in SPLIT_NAMES},
        "actual_no_scaffold": {s: int(split_stats[s]["no_scaffold_count"]) for s in SPLIT_NAMES},
        "actual_scaffold_bearing": {s: int(split_stats[s]["scaffold_bearing_count"]) for s in SPLIT_NAMES},
        "dropped": int(len(dropped_df)),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# =========================
# Worker / launcher
# =========================
def worker_main(args: argparse.Namespace) -> int:
    try:
        if args.condition not in CONDITIONS:
            print(f"[ERROR] Unknown condition: {args.condition}")
            return 2
        csv_path = Path(args.single_csv)
        if not csv_path.exists():
            print(f"[ERROR] Missing CSV: {csv_path}")
            return 2
        condition_name = args.condition
        condition_root = Path(args.output_root) / condition_name
        condition_root.mkdir(parents=True, exist_ok=True)

        summary = process_dataset(
            csv_path=csv_path,
            input_root=Path(args.input_root),
            output_root=condition_root,
            condition_name=condition_name,
            condition_params=CONDITIONS[condition_name],
            smiles_column=args.smiles_column,
            seed=args.seed,
        )
        return 0 if summary is not None else 1
    except Exception:
        print("[WORKER_EXCEPTION]")
        traceback.print_exc()
        return 1


def run_task_subprocess(
    script_path: Path,
    input_root: Path,
    output_root: Path,
    csv_path: Path,
    condition_name: str,
    smiles_column: str,
    seed: int,
    timeout_sec: int,
) -> Dict[str, object]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--single-csv", str(csv_path),
        "--condition", condition_name,
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--smiles-column", smiles_column,
        "--seed", str(seed),
    ]

    rel_path = csv_path.relative_to(input_root)
    log_dir = output_root / "_launcher_logs" / condition_name / rel_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{csv_path.stem}.log"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            errors="replace",
        )
        log_text = []
        log_text.append("$ " + " ".join(cmd))
        log_text.append("\n[STDOUT]\n")
        log_text.append(result.stdout or "")
        log_text.append("\n[STDERR]\n")
        log_text.append(result.stderr or "")
        log_path.write_text("".join(log_text), encoding="utf-8")

        return {
            "ok": result.returncode == 0,
            "returncode": int(result.returncode),
            "error_type": "nonzero_exit" if result.returncode != 0 else "",
            "stdout_tail": safe_tail(result.stdout),
            "stderr_tail": safe_tail(result.stderr),
            "log_path": str(log_path),
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode("utf-8", errors="replace") if e.stdout else "")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode("utf-8", errors="replace") if e.stderr else "")
        log_text = []
        log_text.append("$ " + " ".join(cmd))
        log_text.append(f"\n[TIMEOUT after {timeout_sec}s]\n")
        log_text.append("\n[STDOUT]\n")
        log_text.append(stdout)
        log_text.append("\n[STDERR]\n")
        log_text.append(stderr)
        log_path.write_text("".join(log_text), encoding="utf-8")
        return {
            "ok": False,
            "returncode": -999,
            "error_type": "timeout",
            "stdout_tail": safe_tail(stdout),
            "stderr_tail": safe_tail(stderr),
            "log_path": str(log_path),
        }


def launcher_main(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_csv = find_all_csv_files(input_root)
    if not all_csv:
        print(f"[ERROR] No CSV files found under {input_root}")
        return 1

    script_path = Path(__file__).resolve()
    combined_rows = []
    failed_rows = []

    for condition_name in args.conditions:
        condition_root = output_root / condition_name
        condition_root.mkdir(parents=True, exist_ok=True)
        condition_rows = []

        for csv_path in all_csv:
            print(f"\n[LAUNCH] [{condition_name}] {csv_path}")
            task = run_task_subprocess(
                script_path=script_path,
                input_root=input_root,
                output_root=output_root,
                csv_path=csv_path,
                condition_name=condition_name,
                smiles_column=args.smiles_column,
                seed=args.seed,
                timeout_sec=args.task_timeout_sec,
            )

            if task["ok"]:
                sj = summary_json_path(input_root, condition_root, csv_path)
                if sj.exists():
                    try:
                        row = json.loads(sj.read_text(encoding="utf-8"))
                        condition_rows.append(row)
                        combined_rows.append(row)
                        print(f"[OK] [{condition_name}] {csv_path.stem}")
                    except Exception as e:
                        failed_rows.append({
                            "condition": condition_name,
                            "dataset": csv_path.stem,
                            "csv_path": str(csv_path),
                            "returncode": 0,
                            "error_type": f"summary_read_error: {e}",
                            "log_path": task["log_path"],
                            "stdout_tail": task["stdout_tail"],
                            "stderr_tail": task["stderr_tail"],
                        })
                        print(f"[FAIL] [{condition_name}] {csv_path.stem}: summary read error")
                else:
                    failed_rows.append({
                        "condition": condition_name,
                        "dataset": csv_path.stem,
                        "csv_path": str(csv_path),
                        "returncode": 0,
                        "error_type": "missing_summary_json",
                        "log_path": task["log_path"],
                        "stdout_tail": task["stdout_tail"],
                        "stderr_tail": task["stderr_tail"],
                    })
                    print(f"[FAIL] [{condition_name}] {csv_path.stem}: missing summary.json")
            else:
                failed_rows.append({
                    "condition": condition_name,
                    "dataset": csv_path.stem,
                    "csv_path": str(csv_path),
                    "returncode": int(task["returncode"]),
                    "error_type": task["error_type"],
                    "log_path": task["log_path"],
                    "stdout_tail": task["stdout_tail"],
                    "stderr_tail": task["stderr_tail"],
                })
                print(f"[FAIL] [{condition_name}] {csv_path.stem}: {task['error_type']} (returncode={task['returncode']})")

        if condition_rows:
            write_dict_rows_csv(condition_rows, condition_root / "condition_summary.csv")
            (condition_root / "condition_params.json").write_text(
                json.dumps(CONDITIONS[condition_name], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\n[OK] {condition_name} summary saved to {condition_root / 'condition_summary.csv'}")
        else:
            print(f"\n[WARN] {condition_name}: no dataset completed successfully.")

    if combined_rows:
        write_dict_rows_csv(combined_rows, output_root / "combined_condition_summary.csv")
        print(f"\n[OK] Combined summary saved to {output_root / 'combined_condition_summary.csv'}")
    else:
        print("\n[WARN] No dataset completed successfully for any condition.")

    failed_csv = output_root / "failed_tasks.csv"
    failed_json = output_root / "failed_tasks.json"
    write_dict_rows_csv(failed_rows, failed_csv)
    failed_json.write_text(json.dumps(failed_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print_failed_rows(failed_rows)

    return 0


# =========================
# CLI
# =========================
def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed conditions A and D for the DataSAIL + ESA coupled greedy splitter (resilient subprocess launcher).")
    parser.add_argument("--input-root", default="data/processed", help="Root directory containing processed dataset CSV files.")
    parser.add_argument("--output-root", default="data/splits/esa", help="Output root directory for ESA split files.")
    parser.add_argument("--smiles-column", default=SMILES_COLUMN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["A_balanced_baseline", "D_train_support_priority"],
        default=["A_balanced_baseline", "D_train_support_priority"],
        help="Which fixed conditions to run. Default: both A and D.",
    )
    parser.add_argument("--task-timeout-sec", type=int, default=7200, help="Per-task timeout for subprocess execution.")

    # worker-only args
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-csv", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--condition", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        return worker_main(args)
    return launcher_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
