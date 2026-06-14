from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import DataStructs

from .chem_utils import canonicalize_smiles, get_scaffold, is_no_scaffold, morgan_fp_bitvect


@dataclass(frozen=True)
class LeakageMetrics:
    n_query: int
    n_ref: int
    mean_maxsim: float
    leakrate_at_threshold: float
    L_norm: float

    def as_dict(self) -> dict:
        return {
            "n_query": int(self.n_query),
            "n_ref": int(self.n_ref),
            "mean_maxsim": float(self.mean_maxsim) if np.isfinite(self.mean_maxsim) else np.nan,
            "leakrate_at_threshold": float(self.leakrate_at_threshold) if np.isfinite(self.leakrate_at_threshold) else np.nan,
            "L_norm": float(self.L_norm) if np.isfinite(self.L_norm) else np.nan,
        }


def leakage_metrics(query_fps: Iterable, ref_fps: Iterable, threshold: float = 0.4) -> LeakageMetrics:
    """Compute train-facing similarity exposure metrics.

    Metrics:
    - mean maximum Tanimoto similarity from each query molecule to the reference set;
    - leak rate: fraction of query molecules with max similarity greater than `threshold`;
    - L_norm: average pairwise Tanimoto similarity across query x reference pairs.
    """
    query = [fp for fp in query_fps if fp is not None]
    ref = [fp for fp in ref_fps if fp is not None]
    if not query or not ref:
        return LeakageMetrics(len(query), len(ref), np.nan, np.nan, np.nan)

    max_sims = []
    pair_sum = 0.0
    pair_count = 0
    for fp in query:
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref)
        if not sims:
            continue
        max_sims.append(float(max(sims)))
        pair_sum += float(sum(sims))
        pair_count += len(sims)

    if not max_sims or pair_count == 0:
        return LeakageMetrics(len(query), len(ref), np.nan, np.nan, np.nan)

    max_arr = np.asarray(max_sims, dtype=float)
    return LeakageMetrics(
        n_query=len(query),
        n_ref=len(ref),
        mean_maxsim=float(np.mean(max_arr)),
        leakrate_at_threshold=float(np.mean(max_arr > threshold)),
        L_norm=float(pair_sum / pair_count),
    )


def load_split_csv(path: Path, smiles_column: str = "molecules") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[smiles_column, "canonical_smiles", "scaffold", "no_scaffold", "fp"])
    df = pd.read_csv(path)
    if smiles_column not in df.columns:
        raise ValueError(f"{path} is missing SMILES column '{smiles_column}'")
    out = df.dropna(subset=[smiles_column]).copy()
    out["canonical_smiles"] = out[smiles_column].astype(str).map(canonicalize_smiles)
    out = out.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    out["scaffold"] = out["canonical_smiles"].map(get_scaffold)
    out["no_scaffold"] = out["scaffold"].map(is_no_scaffold)
    out["fp"] = out["canonical_smiles"].map(morgan_fp_bitvect)
    out = out[out["fp"].notna()].reset_index(drop=True)
    return out


def find_split_dirs(root: Path):
    dirs = []
    for test_csv in root.rglob("test.csv"):
        ds_dir = test_csv.parent
        if (ds_dir / "train.csv").exists():
            dirs.append(ds_dir)
    return sorted(set(dirs))


def summarize_split_leakage(split_dir: Path, smiles_column: str = "molecules", threshold: float = 0.4, include_valid: bool = False) -> dict:
    train_df = load_split_csv(split_dir / "train.csv", smiles_column)
    valid_df = load_split_csv(split_dir / "valid.csv", smiles_column)
    test_df = load_split_csv(split_dir / "test.csv", smiles_column)

    train_fps = train_df["fp"].tolist()
    row = {
        "dataset": split_dir.name,
        "split_dir": str(split_dir),
        "train_total_fp": int(len(train_df)),
        "valid_total_fp": int(len(valid_df)),
        "test_total_fp": int(len(test_df)),
        "train_no_scaffold_fp": int(train_df["no_scaffold"].sum()) if len(train_df) else 0,
        "valid_no_scaffold_fp": int(valid_df["no_scaffold"].sum()) if len(valid_df) else 0,
        "test_no_scaffold_fp": int(test_df["no_scaffold"].sum()) if len(test_df) else 0,
    }

    test_metrics = leakage_metrics(
        query_fps=test_df.loc[test_df["no_scaffold"], "fp"].tolist(),
        ref_fps=train_fps,
        threshold=threshold,
    )
    row.update({
        "mean_maxsim_test_ns_to_train_all": test_metrics.mean_maxsim,
        f"leakrate_at_{threshold:g}_test_ns_to_train_all": test_metrics.leakrate_at_threshold,
        "L_norm_test_ns_to_train_all": test_metrics.L_norm,
    })

    if include_valid:
        valid_metrics = leakage_metrics(
            query_fps=valid_df.loc[valid_df["no_scaffold"], "fp"].tolist(),
            ref_fps=train_fps,
            threshold=threshold,
        )
        row.update({
            "mean_maxsim_valid_ns_to_train_all": valid_metrics.mean_maxsim,
            f"leakrate_at_{threshold:g}_valid_ns_to_train_all": valid_metrics.leakrate_at_threshold,
            "L_norm_valid_ns_to_train_all": valid_metrics.L_norm,
        })
    return row
