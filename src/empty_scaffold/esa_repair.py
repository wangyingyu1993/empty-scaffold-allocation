"""Compact ESA repair utilities for archived DataSAIL split outputs.

These functions keep scaffold-bearing molecules in their archived DataSAIL
branches and reassign only no-scaffold molecules with the same micro-cluster +
coupled-greedy ESA logic used by ``scripts/analysis/datasail_and_esa_repair.py``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters.rdSimDivPickers import LeaderPicker

rdBase.BlockLogs()

SMILES_COLUMN = "molecules"
SPLIT_NAMES = ["train", "valid", "test"]
NO_SCAFFOLD_LABELS = {"Invalid_SMILES", "No_Scaffold", "Error"}

ESA_FP_RADIUS = 2
ESA_FP_BITS = 2048
ESA_INIT_SIM_THRESHOLD = 0.40
ESA_RECLUSTER_STEP = 0.00
ESA_MAX_SIM_THRESHOLD = 0.40
ESA_MAX_REFINE_DEPTH = 1

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


def mol2smiles(mol):
    try:
        return Chem.MolToSmiles(Chem.rdmolops.RemoveHs(mol))
    except Exception:
        return None


def normalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return mol2smiles(mol)


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


def smiles_to_fp(smiles: str, radius: int = ESA_FP_RADIUS, n_bits: int = ESA_FP_BITS):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    except Exception:
        return None


def get_split_ratio_and_weights(n: int) -> Tuple[Tuple[float, float, float], List[int]]:
    if n <= 1000:
        return (0.6, 0.2, 0.2), [3, 1, 1]
    if n <= 3000:
        return (5.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0), [5, 1, 1]
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


def refine_large_cluster(global_indices: List[int], all_fps: List, sim_threshold: float, max_cluster_size: int, depth: int = 0) -> List[List[int]]:
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

    return sorted(refined_clusters, key=lambda x: (-len(x), x[0] if len(x) > 0 else -1))


def sample_indices(indices: List[int], cap: int, rng: np.random.Generator) -> List[int]:
    if len(indices) <= cap:
        return list(indices)
    return sorted(rng.choice(indices, size=cap, replace=False).tolist())


def estimate_group_to_train_similarity(group_indices: List[int], all_fps: List, train_fps: List, sample_cap: int, rng: np.random.Generator) -> float:
    if len(group_indices) == 0 or len(train_fps) == 0:
        return 0.0
    idx_g = sample_indices(group_indices, sample_cap, rng)
    if len(train_fps) <= sample_cap:
        sampled_train = train_fps
    else:
        sampled_train = [train_fps[i] for i in sorted(rng.choice(len(train_fps), size=sample_cap, replace=False).tolist())]
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


def coupled_assign_no_scaffold(
    df_no_scaffold: pd.DataFrame,
    ratios: Tuple[float, float, float],
    scaffold_bearing_train_fps: List,
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
            cluster_name = "ESA_%05d" % cid
            member_ids = [ids[valid_local_indices[j]] for j in cluster_local]
            for molecule_id in member_ids:
                id_to_cluster_id[molecule_id] = cluster_name
            groups.append({
                "cluster_id": cluster_name,
                "member_ids": member_ids,
                "member_fp_local_idx": list(cluster_local),
                "size": len(member_ids),
                "is_invalid_fp_group": False,
            })

    start_idx = len(groups) + 1
    for offset, local_idx in enumerate(invalid_local_indices):
        cluster_name = "ESA_%05d" % (start_idx + offset)
        molecule_id = ids[local_idx]
        id_to_cluster_id[molecule_id] = cluster_name
        groups.append({
            "cluster_id": cluster_name,
            "member_ids": [molecule_id],
            "member_fp_local_idx": [],
            "size": 1,
            "is_invalid_fp_group": True,
        })

    if not groups:
        empty = pd.DataFrame(columns=["cluster_id", "split", "cluster_size", "const_to_sb_train", "is_invalid_fp_group"])
        return {}, {}, empty

    const_to_sb_train = []
    for group in groups:
        if group["is_invalid_fp_group"] or len(scaffold_bearing_train_fps) == 0:
            const_to_sb_train.append(0.0)
        else:
            const_to_sb_train.append(
                estimate_group_to_train_similarity(group["member_fp_local_idx"], fps_all, scaffold_bearing_train_fps, sample_cap, rng)
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

    def feasible_after(candidate_split: str, group_idx: int, pos: int) -> bool:
        cand_counts = split_counts.copy()
        cand_dropped = dropped_count
        cand_size = groups[group_idx]["size"]
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
        for split in SPLIT_NAMES:
            denom = max(target_counts[split], 1)
            overflow = max(0, after_counts[split] - target_counts[split]) / denom
            deviation = abs(after_counts[split] - target_counts[split]) / denom
            cost += 3.0 * overflow + deviation
        return lambda_imbalance * cost

    def pairwise_increment(candidate_split: str, group_idx: int) -> float:
        cost = 0.0
        if candidate_split == "drop":
            return lambda_drop * groups[group_idx]["size"]
        if candidate_split == "valid":
            cost += lambda_valid_trainall * const_to_sb_train[group_idx]
        elif candidate_split == "test":
            cost += lambda_test_trainall * const_to_sb_train[group_idx]
        for split in SPLIT_NAMES:
            for assigned_idx in assigned_by_split[split]:
                sim = pair_sim[group_idx, assigned_idx]
                if sim <= 0:
                    continue
                if {candidate_split, split} == {"train", "valid"}:
                    cost += (lambda_internal + lambda_valid_trainall) * sim
                elif {candidate_split, split} == {"train", "test"}:
                    cost += (lambda_internal + lambda_test_trainall) * sim
                elif {candidate_split, split} == {"valid", "test"}:
                    cost += lambda_internal * sim
        return cost

    for pos, group_idx in enumerate(order):
        group_size = groups[group_idx]["size"]
        best_split = None
        best_score = None
        for cand in ["train", "valid", "test", "drop"]:
            if not feasible_after(cand, group_idx, pos):
                continue
            after_counts = split_counts.copy()
            if cand != "drop":
                after_counts[cand] += group_size
            deficit_after = sum(max(0, lower_bounds[s] - after_counts[s]) for s in SPLIT_NAMES)
            deficit_penalty = deficit_after / max(1, len(df_no_scaffold))
            unmet_splits = sum(1 for s in SPLIT_NAMES if after_counts[s] < lower_bounds[s])
            drop_extra = 10.0 * unmet_splits if cand == "drop" and unmet_splits > 0 else 0.0
            score = pairwise_increment(cand, group_idx) + imbalance_cost(after_counts) + 2.0 * deficit_penalty + drop_extra
            if best_score is None or score < best_score - 1e-12:
                best_split, best_score = cand, score
            elif abs(score - best_score) <= 1e-12:
                if best_split == "drop" and cand != "drop":
                    best_split, best_score = cand, score
                elif cand != "drop" and best_split != "drop" and split_counts[cand] < split_counts[best_split]:
                    best_split, best_score = cand, score

        if best_split is None:
            ranked = ["train", "valid", "test", "drop"]
            best_split = ranked[0]
            best_score = float("inf")
            for cand in ranked:
                after_counts = split_counts.copy()
                if cand != "drop":
                    after_counts[cand] += group_size
                score = pairwise_increment(cand, group_idx) + imbalance_cost(after_counts)
                if score < best_score:
                    best_split, best_score = cand, score

        assignments[group_idx] = best_split
        if best_split == "drop":
            dropped_count += group_size
        else:
            split_counts[best_split] += group_size
            assigned_by_split[best_split].append(group_idx)

    needy = [s for s in SPLIT_NAMES if split_counts[s] < lower_bounds[s]]
    if needy:
        train_groups = list(assigned_by_split["train"])
        train_groups_sorted = sorted(train_groups, key=lambda i: (groups[i]["size"], const_to_sb_train[i]))
        for need_split in needy:
            while split_counts[need_split] < lower_bounds[need_split] and train_groups_sorted:
                group_idx = train_groups_sorted.pop(0)
                group_size = groups[group_idx]["size"]
                if split_counts["train"] - group_size < lower_bounds["train"]:
                    continue
                assignments[group_idx] = need_split
                split_counts["train"] -= group_size
                split_counts[need_split] += group_size
                if group_idx in assigned_by_split["train"]:
                    assigned_by_split["train"].remove(group_idx)
                assigned_by_split[need_split].append(group_idx)

    id_to_split: Dict[str, str] = {}
    cluster_rows = []
    for i, group in enumerate(groups):
        split = assignments.get(i, "drop")
        for molecule_id in group["member_ids"]:
            id_to_split[molecule_id] = split
        cluster_rows.append({
            "cluster_id": group["cluster_id"],
            "split": split,
            "cluster_size": group["size"],
            "const_to_sb_train": float(const_to_sb_train[i]),
            "is_invalid_fp_group": bool(group["is_invalid_fp_group"]),
        })

    return id_to_split, id_to_cluster_id, pd.DataFrame(cluster_rows)


def load_archived_datasail_split(split_dir: Path, smiles_column: str = SMILES_COLUMN) -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    original_columns = None
    dataset = Path(split_dir).name
    for split in SPLIT_NAMES:
        path = Path(split_dir) / (split + ".csv")
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if smiles_column not in df.columns:
            raise ValueError("%s missing smiles column %s" % (path, smiles_column))
        if original_columns is None:
            original_columns = list(df.columns)
        df = df.copy()
        df["RawSplit"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=[smiles_column]).copy().reset_index(drop=True)
    full["SMILES_norm"] = full[smiles_column].astype(str).map(normalize_smiles)
    full = full.dropna(subset=["SMILES_norm"]).reset_index(drop=True)
    full["Scaffold"] = full["SMILES_norm"].astype(str).map(get_scaffold)
    full["Is_NoScaffold"] = full["Scaffold"].isin(NO_SCAFFOLD_LABELS)
    full["ID"] = ["%s_%06d" % (dataset, i + 1) for i in range(len(full))]
    return full, original_columns or []


def repair_archived_datasail_split(
    split_dir: Path,
    condition_name: str = "D_train_support_priority",
    smiles_column: str = SMILES_COLUMN,
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    if condition_name not in CONDITIONS:
        raise ValueError("Unknown ESA condition: %s" % condition_name)
    condition = CONDITIONS[condition_name]
    full, original_columns = load_archived_datasail_split(split_dir, smiles_column=smiles_column)
    ratios, _ = get_split_ratio_and_weights(len(full))

    scaffold_df = full[~full["Is_NoScaffold"]].copy()
    no_scaffold_df = full[full["Is_NoScaffold"]].copy().reset_index(drop=True)

    scaffold_bearing_train_fps = []
    for _, row in scaffold_df[scaffold_df["RawSplit"] == "train"].iterrows():
        fp = smiles_to_fp(row["SMILES_norm"])
        if fp is not None:
            scaffold_bearing_train_fps.append(fp)

    ns_id_to_split, ns_id_to_cluster, cluster_summary = coupled_assign_no_scaffold(
        df_no_scaffold=no_scaffold_df,
        ratios=ratios,
        scaffold_bearing_train_fps=scaffold_bearing_train_fps,
        seed=seed,
        sample_cap=int(condition["sample_cap"]),
        lambda_internal=float(condition["lambda_internal"]),
        lambda_valid_trainall=float(condition["lambda_valid_trainall"]),
        lambda_test_trainall=float(condition["lambda_test_trainall"]),
        lambda_imbalance=float(condition["lambda_imbalance"]),
        lambda_drop=float(condition["lambda_drop"]),
        lower_train_frac=float(condition["lower_train_frac"]),
        lower_valid_frac=float(condition["lower_valid_frac"]),
        lower_test_frac=float(condition["lower_test_frac"]),
        drop_budget_frac=float(condition["drop_budget_frac"]),
        esa_max_cluster_to_smallest_target=float(condition["esa_max_cluster_to_smallest_target"]),
    )

    repaired = full.copy()
    repaired["Split"] = repaired["RawSplit"]
    repaired.loc[repaired["Is_NoScaffold"], "Split"] = repaired.loc[repaired["Is_NoScaffold"], "ID"].map(ns_id_to_split)
    repaired["SplitSource"] = np.where(repaired["Is_NoScaffold"], "ESA_" + condition_name, "DataSAIL_scaffold_backbone")
    repaired["ESA_cluster_id"] = repaired["ID"].map(ns_id_to_cluster)

    split_tables: Dict[str, pd.DataFrame] = {}
    for split in SPLIT_NAMES:
        split_tables[split] = repaired[repaired["Split"] == split][original_columns].copy().reset_index(drop=True)
    if (repaired["Split"] == "drop").any():
        split_tables["drop"] = repaired[repaired["Split"] == "drop"][original_columns].copy().reset_index(drop=True)

    audit_cols = ["ID", smiles_column, "SMILES_norm", "Scaffold", "Is_NoScaffold", "RawSplit", "Split", "SplitSource", "ESA_cluster_id"]
    return split_tables, repaired[audit_cols].copy(), cluster_summary


def write_repaired_split_tables(split_tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, df in split_tables.items():
        df.to_csv(output_dir / (split + ".csv"), index=False)
