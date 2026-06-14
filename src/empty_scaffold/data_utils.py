from __future__ import annotations

from pathlib import Path

import pandas as pd

from .chem_utils import canonicalize_smiles, get_scaffold, is_invalid_scaffold, is_no_scaffold


def annotate_scaffolds(df: pd.DataFrame, smiles_col: str = "molecules") -> pd.DataFrame:
    out = df.copy()
    out["canonical_smiles"] = out[smiles_col].apply(canonicalize_smiles)
    out = out.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    out["scaffold"] = out["canonical_smiles"].apply(get_scaffold)
    out = out.loc[~out["scaffold"].apply(is_invalid_scaffold)].reset_index(drop=True)
    out["no_scaffold"] = out["scaffold"].apply(is_no_scaffold)
    return out


def summarize_scaffolds(df: pd.DataFrame, dataset: str) -> dict:
    n_total = int(len(df))
    n_ns = int(df["no_scaffold"].sum()) if "no_scaffold" in df.columns else 0
    return {
        "dataset": dataset,
        "n_molecules": n_total,
        "n_no_scaffold": n_ns,
        "no_scaffold_proportion": n_ns / n_total if n_total else float("nan"),
        "n_unique_nonempty_scaffolds": int(df.loc[~df["no_scaffold"], "scaffold"].nunique()) if n_total else 0,
    }


def summarize_split(split_dir: Path, dataset: str, smiles_col: str = "molecules") -> pd.DataFrame:
    rows = []
    for split in ["train", "valid", "test"]:
        path = split_dir / f"{split}.csv"
        df = pd.read_csv(path)
        ann = annotate_scaffolds(df, smiles_col)
        summary = summarize_scaffolds(ann, dataset)
        summary["split"] = split
        summary["n_rows_original"] = int(len(df))
        rows.append(summary)
    full = pd.DataFrame(rows)
    overall_n = int(full["n_molecules"].sum())
    overall_ns = int(full["n_no_scaffold"].sum())
    overall_prop = overall_ns / overall_n if overall_n else float("nan")
    full["overall_no_scaffold_proportion"] = overall_prop
    full["support_recovery_vs_overall"] = full["no_scaffold_proportion"] / overall_prop if overall_prop else float("nan")
    return full
