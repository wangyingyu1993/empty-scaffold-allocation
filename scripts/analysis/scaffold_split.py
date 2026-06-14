#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Bemis-Murcko scaffold splits.

Input layout:
  data/processed/
    admet/classification/*.csv
    admet/regression/*.csv
    moleculenet/classification/*.csv
    moleculenet/regression/*.csv

Output layout:
  data/splits/scaffold/
    <benchmark>/<task_group>/<dataset>/fold_<n>/train.csv
    <benchmark>/<task_group>/<dataset>/fold_<n>/valid.csv
    <benchmark>/<task_group>/<dataset>/fold_<n>/test.csv

Molecules with empty Bemis-Murcko scaffold representations are assigned to one
No_Scaffold bucket.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

rdBase.BlockLogs()

SMILES_COLUMN = "molecules"
SEED = 10086
NO_SCAFFOLD_LABEL = "No_Scaffold"
INVALID_SCAFFOLD_LABELS = {"Invalid_SMILES", "Error"}
SPLIT_ORDER = ("train", "valid", "test")


def get_split_weights_and_kfolds(n: int) -> Tuple[List[int], int]:
    """Return integer train/valid/test slot weights and number of folds."""
    if n <= 1000:
        return [3, 1, 1], 5
    if n <= 3000:
        return [5, 1, 1], 7
    return [8, 1, 1], 10


def find_csv_files(root: Path) -> List[Path]:
    files = []
    for benchmark in ("admet", "moleculenet"):
        for task_group in ("classification", "regression"):
            folder = root / benchmark / task_group
            if not folder.exists():
                print(f"[WARN] Missing folder: {folder}")
                continue
            files.extend(sorted(p for p in folder.glob("*.csv") if p.is_file()))
    return sorted(files)


def infer_benchmark_task_dataset(csv_path: Path, input_root: Path) -> Tuple[str, str, str]:
    rel = csv_path.relative_to(input_root)
    benchmark = rel.parts[0] if len(rel.parts) >= 1 else ""
    task_type = rel.parts[1] if len(rel.parts) >= 2 else ""
    dataset = csv_path.stem
    return benchmark, task_type, dataset


def build_output_dir(csv_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = csv_path.relative_to(input_root)
    out_dir = output_root / rel.parent / csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        mol = Chem.rdmolops.RemoveHs(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_bemis_murcko_scaffold(canonical_smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(canonical_smiles))
    if mol is None:
        return "Invalid_SMILES"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return NO_SCAFFOLD_LABEL
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return "Invalid_SMILES"


def summarize_split(df: pd.DataFrame) -> Dict[str, object]:
    n = len(df)
    if n == 0:
        return {
            "molecule_count": 0,
            "scaffold_bearing_count": 0,
            "no_scaffold_count": 0,
            "no_scaffold_proportion": 0.0,
            "no_scaffold_percentage": 0.0,
            "scaffold_group_count": 0,
        }
    no_mask = df["Scaffold"].eq(NO_SCAFFOLD_LABEL)
    no_count = int(no_mask.sum())
    scaffold_bearing_count = int(n - no_count)
    scaffold_group_count = int(df.loc[~no_mask, "Scaffold"].nunique())
    return {
        "molecule_count": int(n),
        "scaffold_bearing_count": scaffold_bearing_count,
        "no_scaffold_count": no_count,
        "no_scaffold_proportion": float(no_count / n),
        "no_scaffold_percentage": float(100.0 * no_count / n),
        "scaffold_group_count": scaffold_group_count,
    }


def assign_groups_to_parts(
    group_to_rowids: Dict[str, List[str]],
    kfolds: int,
    seed: int,
) -> Dict[str, int]:
    """Greedy group-to-part assignment, preserving each scaffold as indivisible."""
    if len(group_to_rowids) < kfolds:
        raise RuntimeError(
            f"Number of scaffold groups ({len(group_to_rowids)}) is smaller than kfolds={kfolds}; "
            "cannot construct non-empty scaffold-preserving parts."
        )

    rng = np.random.RandomState(seed)
    items = list(group_to_rowids.items())
    rng.shuffle(items)
    items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)

    part_sizes = {i: 0 for i in range(kfolds)}
    rowid_to_part: Dict[str, int] = {}

    for part_id, (_, rowids) in enumerate(items[:kfolds]):
        for rid in rowids:
            rowid_to_part[rid] = part_id
        part_sizes[part_id] += len(rowids)

    for _, rowids in items[kfolds:]:
        min_size = min(part_sizes.values())
        candidate_parts = [p for p, size in part_sizes.items() if size == min_size]
        part_id = int(rng.choice(candidate_parts))
        for rid in rowids:
            rowid_to_part[rid] = part_id
        part_sizes[part_id] += len(rowids)

    perm = np.arange(kfolds)
    rng.shuffle(perm)
    old_to_new = {old: int(new) for old, new in zip(range(kfolds), perm)}
    return {rid: old_to_new[part] for rid, part in rowid_to_part.items()}


def process_one_dataset(
    csv_path: Path,
    input_root: Path,
    output_root: Path,
    smiles_column: str,
    seed: int,
    save_part_files: bool,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    benchmark, task_type, dataset = infer_benchmark_task_dataset(csv_path, input_root)
    out_dir = build_output_dir(csv_path, input_root, output_root)

    raw = pd.read_csv(csv_path)
    if smiles_column not in raw.columns:
        raise KeyError(f"{csv_path} is missing SMILES column: {smiles_column}")

    original_columns = raw.columns.tolist()
    raw_total = len(raw)

    df = raw.dropna(subset=[smiles_column]).copy().reset_index(drop=True)
    after_dropna = len(df)

    df["SMILES_norm"] = df[smiles_column].map(canonicalize_smiles)
    df = df.dropna(subset=["SMILES_norm"]).copy().reset_index(drop=True)
    effective_total = len(df)
    if effective_total == 0:
        raise ValueError(f"{dataset}: no valid molecules after SMILES parsing.")

    df["Scaffold"] = df["SMILES_norm"].map(get_bemis_murcko_scaffold)
    df = df.loc[~df["Scaffold"].isin(INVALID_SCAFFOLD_LABELS)].copy().reset_index(drop=True)
    effective_total = len(df)
    if effective_total == 0:
        raise ValueError(f"{dataset}: no valid molecules after scaffold extraction.")

    df["ROW_ID"] = [f"{dataset}_{i + 1:08d}" for i in range(effective_total)]

    weights, kfolds = get_split_weights_and_kfolds(effective_total)
    train_slots, valid_slots, test_slots = weights
    assert valid_slots == 1 and test_slots == 1, "Sliding-window implementation expects one valid and one test slot."

    group_to_rowids = df.groupby("Scaffold")["ROW_ID"].apply(list).to_dict()
    rowid_to_part = assign_groups_to_parts(group_to_rowids, kfolds=kfolds, seed=seed)

    df["Part_ID"] = df["ROW_ID"].map(rowid_to_part).astype(int)
    df["Part_Name"] = df["Part_ID"].map(lambda x: f"part_{x + 1}")

    if df["Part_ID"].nunique() != kfolds:
        raise RuntimeError(f"{dataset}: not all {kfolds} parts are non-empty.")

    audit_cols = ["ROW_ID", smiles_column, "SMILES_norm", "Scaffold", "Part_ID", "Part_Name"]
    df[audit_cols].to_csv(out_dir / "split_assignment_audit.csv", index=False, encoding="utf-8-sig")

    part_summary_rows: List[dict] = []
    for part_id in range(kfolds):
        part_name = f"part_{part_id + 1}"
        part_df = df[df["Part_ID"] == part_id].copy().reset_index(drop=True)
        stats = summarize_split(part_df)
        part_summary_rows.append({
            "benchmark": benchmark,
            "task_type": task_type,
            "dataset": dataset,
            "splitter": "scaffold",
            "kfolds": kfolds,
            "part_order": part_id + 1,
            "part_name": part_name,
            **stats,
        })
        if save_part_files:
            part_df[original_columns].to_csv(out_dir / f"{part_name}.csv", index=False, encoding="utf-8-sig")

    fold_long_rows: List[dict] = []
    fold_wide_rows: List[dict] = []

    for fold in range(kfolds):
        fold_dir = out_dir / f"fold_{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_part = fold
        valid_part = (fold + 1) % kfolds
        train_parts = [p for p in range(kfolds) if p not in {valid_part, test_part}]

        split_frames = {
            "train": df[df["Part_ID"].isin(train_parts)].sample(frac=1, random_state=seed + 1000 + fold).reset_index(drop=True),
            "valid": df[df["Part_ID"] == valid_part].sample(frac=1, random_state=seed + 2000 + fold).reset_index(drop=True),
            "test": df[df["Part_ID"] == test_part].sample(frac=1, random_state=seed + 3000 + fold).reset_index(drop=True),
        }

        for split_name, split_df in split_frames.items():
            split_df[original_columns].to_csv(fold_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")
            stats = summarize_split(split_df)
            fold_long_rows.append({
                "benchmark": benchmark,
                "task_type": task_type,
                "dataset": dataset,
                "splitter": "scaffold",
                "fold": fold + 1,
                "split": split_name,
                **stats,
            })

        wide = {
            "benchmark": benchmark,
            "task_type": task_type,
            "dataset": dataset,
            "splitter": "scaffold",
            "fold": fold + 1,
            "kfolds": kfolds,
            "test_part": test_part + 1,
            "valid_part": valid_part + 1,
        }
        for split_name, split_df in split_frames.items():
            stats = summarize_split(split_df)
            for key, value in stats.items():
                wide[f"{split_name}_{key}"] = value
        fold_wide_rows.append(wide)

    dataset_stats = summarize_split(df)
    dataset_row = {
        "benchmark": benchmark,
        "task_type": task_type,
        "dataset": dataset,
        "splitter": "scaffold",
        "status": "success",
        "raw_total_count": raw_total,
        "after_dropna_count": after_dropna,
        "effective_total_count": effective_total,
        "removed_invalid_or_empty_smiles_count": raw_total - effective_total,
        "kfolds": kfolds,
        "weight_splits": ":".join(map(str, weights)),
        **dataset_stats,
    }
    return fold_long_rows, fold_wide_rows, part_summary_rows, dataset_row


def sort_summary(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(columns).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bemis-Murcko scaffold split folds.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing input CSV files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for split files and summaries.")
    parser.add_argument("--smiles-column", default=SMILES_COLUMN, help="SMILES column name.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--save-part-files", action="store_true", help="Also save part_i.csv files for each dataset.")
    args = parser.parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    csv_files = find_csv_files(args.input_root)
    if not csv_files:
        raise RuntimeError(f"No CSV files found under {args.input_root}")

    dataset_rows: List[dict] = []
    fold_long_rows: List[dict] = []
    fold_wide_rows: List[dict] = []
    part_rows: List[dict] = []
    failed_rows: List[dict] = []

    for csv_path in csv_files:
        try:
            fold_long, fold_wide, part_summary, dataset_row = process_one_dataset(
                csv_path=csv_path,
                input_root=args.input_root,
                output_root=args.output_root,
                smiles_column=args.smiles_column,
                seed=args.seed,
                save_part_files=args.save_part_files,
            )
            fold_long_rows.extend(fold_long)
            fold_wide_rows.extend(fold_wide)
            part_rows.extend(part_summary)
            dataset_rows.append(dataset_row)
            print(f"[OK] {csv_path.relative_to(args.input_root)}")
        except Exception as exc:
            benchmark, task_type, dataset = infer_benchmark_task_dataset(csv_path, args.input_root)
            failed_rows.append({
                "benchmark": benchmark,
                "task_type": task_type,
                "dataset": dataset,
                "path": str(csv_path),
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            })
            dataset_rows.append({
                "benchmark": benchmark,
                "task_type": task_type,
                "dataset": dataset,
                "splitter": "scaffold",
                "status": "failed",
                "error_message": str(exc),
            })
            print(f"[FAILED] {csv_path.relative_to(args.input_root)}: {exc}")

    dataset_df = sort_summary(pd.DataFrame(dataset_rows), ["benchmark", "task_type", "dataset"])
    fold_long_df = sort_summary(pd.DataFrame(fold_long_rows), ["benchmark", "task_type", "dataset", "fold", "split"])
    fold_wide_df = sort_summary(pd.DataFrame(fold_wide_rows), ["benchmark", "task_type", "dataset", "fold"])
    part_df = sort_summary(pd.DataFrame(part_rows), ["benchmark", "task_type", "dataset", "part_order"])

    dataset_df.to_csv(args.output_root / "scaffold_dataset_summary.csv", index=False, encoding="utf-8-sig")
    fold_long_df.to_csv(args.output_root / "scaffold_fold_summary_long.csv", index=False, encoding="utf-8-sig")
    fold_wide_df.to_csv(args.output_root / "scaffold_fold_summary_wide.csv", index=False, encoding="utf-8-sig")
    part_df.to_csv(args.output_root / "scaffold_part_summary.csv", index=False, encoding="utf-8-sig")

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(args.output_root / "scaffold_failed.csv", index=False, encoding="utf-8-sig")

    print(f"Done. Outputs saved to: {args.output_root}")


if __name__ == "__main__":
    main()
