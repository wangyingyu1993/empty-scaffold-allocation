#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate UMAP clustering splits.

Input layout:
  data/processed/
    admet/classification/*.csv
    admet/regression/*.csv
    moleculenet/classification/*.csv
    moleculenet/regression/*.csv

Output layout:
  data/splits/umap/
    <benchmark>/<task_group>/<dataset>/fold_<n>/train.csv
    <benchmark>/<task_group>/<dataset>/fold_<n>/valid.csv
    <benchmark>/<task_group>/<dataset>/fold_<n>/test.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.cluster import AgglomerativeClustering

try:
    import umap
except ImportError as exc:
    raise SystemExit("UMAP is required. Install with: pip install umap-learn") from exc

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")

RANDOM_SEED = 20250328
FP_RADIUS = 2
FP_NBITS = 1024


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{RANDOM_SEED}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def collect_input_files(input_root: Path) -> pd.DataFrame:
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

    return pd.DataFrame(records)


def split_weights(n_molecules: int):
    if n_molecules <= 1000:
        return 3, 1, 1
    if n_molecules <= 3000:
        return 5, 1, 1
    return 8, 1, 1


def mol_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    except Exception:
        return None


def fingerprint_matrix(fps):
    arr = np.zeros((len(fps), FP_NBITS), dtype=np.uint8)
    for i, fp in enumerate(fps):
        on_bits = list(fp.GetOnBits())
        arr[i, on_bits] = 1
    return arr


def assign_clusters_to_bins(cluster_labels, n_bins: int):
    cluster_to_indices = {}
    for idx, label in enumerate(cluster_labels):
        cluster_to_indices.setdefault(int(label), []).append(idx)

    clusters = sorted(cluster_to_indices.items(), key=lambda x: (-len(x[1]), x[0]))
    bins = [[] for _ in range(n_bins)]
    bin_sizes = [0 for _ in range(n_bins)]

    for _, indices in clusters:
        target = int(np.argmin(bin_sizes))
        bins[target].extend(indices)
        bin_sizes[target] += len(indices)

    return bins, cluster_to_indices


def make_sliding_window_splits(df: pd.DataFrame, bins, weights):
    train_w, valid_w, test_w = weights
    n_bins = train_w + valid_w + test_w

    splits = []
    for fold_id in range(n_bins):
        test_bins = [(fold_id + i) % n_bins for i in range(test_w)]
        valid_bins = [(fold_id + test_w + i) % n_bins for i in range(valid_w)]
        train_bins = [i for i in range(n_bins) if i not in set(test_bins + valid_bins)]

        test_idx = sorted(i for b in test_bins for i in bins[b])
        valid_idx = sorted(i for b in valid_bins for i in bins[b])
        train_idx = sorted(i for b in train_bins for i in bins[b])

        splits.append({
            "fold": fold_id + 1,
            "train": df.iloc[train_idx].copy(),
            "valid": df.iloc[valid_idx].copy(),
            "test": df.iloc[test_idx].copy(),
            "train_bins": train_bins,
            "valid_bins": valid_bins,
            "test_bins": test_bins,
        })

    return splits


def write_split_outputs(output_root: Path, benchmark: str, task_group: str, dataset: str, splits):
    rows = []

    for split in splits:
        fold_dir = output_root / benchmark / task_group / dataset / f"fold_{split['fold']}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for branch in ["train", "valid", "test"]:
            split[branch].to_csv(fold_dir / f"{branch}.csv", index=False, encoding="utf-8-sig")
            rows.append({
                "Benchmark": benchmark,
                "TaskGroup": task_group,
                "Dataset": dataset,
                "Fold": split["fold"],
                "Branch": branch,
                "N": int(len(split[branch])),
            })

    return rows


def process_dataset(csv_path: Path, output_root: Path, benchmark: str, task_group: str, dataset: str):
    df = pd.read_csv(csv_path)
    if "molecules" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a molecules column")

    df = df.copy()
    df["molecules"] = df["molecules"].astype(str)
    df["Fingerprint"] = df["molecules"].map(mol_fingerprint)
    df = df[df["Fingerprint"].notna()].reset_index(drop=True)

    weights = split_weights(len(df))
    n_bins = sum(weights)

    x = fingerprint_matrix(df["Fingerprint"].tolist())

    reducer = umap.UMAP(
        n_neighbors=100,
        min_dist=0.0,
        n_components=2,
        metric="jaccard",
        random_state=stable_seed(f"{benchmark}:{task_group}:{dataset}:umap"),
    )
    embedding = reducer.fit_transform(x)

    clustering = AgglomerativeClustering(n_clusters=n_bins, linkage="ward")
    cluster_labels = clustering.fit_predict(embedding)

    bins, cluster_to_indices = assign_clusters_to_bins(cluster_labels, n_bins)
    splits = make_sliding_window_splits(df.drop(columns=["Fingerprint"]), bins, weights)
    branch_summary = write_split_outputs(output_root, benchmark, task_group, dataset, splits)

    cluster_rows = []
    for cluster_id, indices in sorted(cluster_to_indices.items()):
        cluster_rows.append({
            "Benchmark": benchmark,
            "TaskGroup": task_group,
            "Dataset": dataset,
            "Cluster": int(cluster_id),
            "N": int(len(indices)),
        })

    cluster_table = pd.DataFrame(cluster_rows)
    dataset_dir = output_root / benchmark / task_group / dataset
    cluster_table.to_csv(dataset_dir / "umap_cluster_summary.csv", index=False, encoding="utf-8-sig")

    return {
        "Benchmark": benchmark,
        "TaskGroup": task_group,
        "Dataset": dataset,
        "Total_Molecules": int(len(df)),
        "Clusters": int(n_bins),
        "Train_Weight": weights[0],
        "Valid_Weight": weights[1],
        "Test_Weight": weights[2],
        "NFolds": int(n_bins),
        "BranchSummary": branch_summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate UMAP clustering splits.")
    parser.add_argument("--input-root", type=Path, default=Path("data/processed"), help="Processed dataset directory.")
    parser.add_argument("--output-root", type=Path, default=Path("data/splits/umap"), help="UMAP split output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    dataset_files = collect_input_files(args.input_root)
    if dataset_files.empty:
        raise SystemExit(f"No dataset CSV files found under {args.input_root}")

    dataset_summaries = []
    branch_summaries = []

    for _, item in dataset_files.iterrows():
        summary = process_dataset(
            csv_path=item["CSV_Path"],
            output_root=args.output_root,
            benchmark=item["Benchmark"],
            task_group=item["TaskGroup"],
            dataset=item["Dataset"],
        )

        branch_summaries.extend(summary.pop("BranchSummary"))
        dataset_summaries.append(summary)

        print(f"Generated UMAP split: {item['Benchmark']}/{item['TaskGroup']}/{item['Dataset']}")

    pd.DataFrame(dataset_summaries).to_csv(
        args.output_root / "umap_split_dataset_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(branch_summaries).to_csv(
        args.output_root / "umap_split_branch_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "random_seed": RANDOM_SEED,
        "fingerprint": {"type": "Morgan", "radius": FP_RADIUS, "n_bits": FP_NBITS},
        "umap": {"n_neighbors": 100, "min_dist": 0.0, "n_components": 2, "metric": "jaccard"},
        "clustering": {"method": "AgglomerativeClustering", "linkage": "ward"},
        "split_weights": {
            "n <= 1000": [3, 1, 1],
            "1000 < n <= 3000": [5, 1, 1],
            "n > 3000": [8, 1, 1],
        },
    }
    (args.output_root / "umap_split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Processed datasets: {len(dataset_summaries)}")
    print(f"Output directory: {args.output_root}")


if __name__ == "__main__":
    main()
