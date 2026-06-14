from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .data_utils import annotate_scaffolds


DEFAULT_SEED = 10086


def scaffold_kfolds_for_n(n: int) -> int:
    if n <= 1000:
        return 5
    if n <= 3000:
        return 7
    return 10


@dataclass(frozen=True)
class ScaffoldFoldSummary:
    dataset: str
    fold: int
    n_total: int
    n_no_scaffold: int
    no_scaffold_proportion: float


def assign_scaffold_folds(
    df: pd.DataFrame,
    *,
    smiles_col: str = "molecules",
    n_folds: int = 5,
    dataset: Optional[str] = None,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Assign whole Bemis-Murcko scaffold groups to folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if smiles_col not in df.columns:
        raise ValueError(f"missing SMILES column: {smiles_col}")

    out = annotate_scaffolds(df, smiles_col=smiles_col).reset_index(drop=True)
    out["scaffold_group"] = np.where(out["no_scaffold"], "__NO_SCAFFOLD__", out["scaffold"])

    rng = np.random.RandomState(seed)
    groups = []
    for group, idx in out.groupby("scaffold_group", sort=False).groups.items():
        groups.append((str(group), list(idx)))

    if len(groups) < n_folds:
        raise ValueError(f"number of scaffold groups ({len(groups)}) is smaller than n_folds={n_folds}")

    rng.shuffle(groups)
    groups = sorted(groups, key=lambda item: len(item[1]), reverse=True)

    fold_sizes = [0 for _ in range(n_folds)]
    fold_ids = np.empty(len(out), dtype=int)

    for fold, (_, idx) in enumerate(groups[:n_folds]):
        fold_ids[idx] = fold
        fold_sizes[fold] += len(idx)

    for _, idx in groups[n_folds:]:
        min_size = min(fold_sizes)
        candidates = [i for i, size in enumerate(fold_sizes) if size == min_size]
        fold = int(rng.choice(candidates))
        fold_ids[idx] = fold
        fold_sizes[fold] += len(idx)

    perm = np.arange(n_folds)
    rng.shuffle(perm)
    remap = {old: int(new) for old, new in zip(range(n_folds), perm)}
    out["fold"] = np.asarray([remap[int(f)] for f in fold_ids], dtype=int)
    if dataset is not None:
        out["dataset"] = dataset
    return out


def summarize_fold_allocation(folded: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for fold, part in folded.groupby("fold", sort=True):
        n_total = int(len(part))
        n_ns = int(part["no_scaffold"].sum())
        rows.append({
            "dataset": dataset,
            "fold": int(fold),
            "n_total": n_total,
            "n_no_scaffold": n_ns,
            "no_scaffold_proportion": n_ns / n_total if n_total else np.nan,
        })
    return pd.DataFrame(rows)


def no_scaffold_fold(folded: pd.DataFrame) -> int:
    counts = folded.groupby("fold")["no_scaffold"].sum()
    if counts.empty:
        raise ValueError("no folds found")
    return int(counts.idxmax())


def train_test_for_fold(folded: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = folded.loc[folded["fold"] != fold].copy().reset_index(drop=True)
    test = folded.loc[folded["fold"] == fold].copy().reset_index(drop=True)
    return train, test
