#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute no-scaffold-to-train leakage/exposure metrics for split folders.

Expected split layout, recursively under --split-root:

    .../<dataset>/train.csv
    .../<dataset>/valid.csv
    .../<dataset>/test.csv

For each dataset directory, the default query set is test no-scaffold molecules
and the reference set is all training molecules. With --include-valid the same
metrics are also computed for valid no-scaffold molecules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from empty_scaffold.leakage import find_split_dirs, summarize_split_leakage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute no-scaffold leakage metrics from split directories.")
    parser.add_argument(
        "--split-root",
        "--output-root",
        dest="split_root",
        default="data/splits/esa",
        help="Root directory containing per-dataset split folders.",
    )
    parser.add_argument("--smiles-column", default="molecules")
    parser.add_argument("--threshold", type=float, default=0.4, help="Threshold for LeakRate@threshold.")
    parser.add_argument("--include-valid", action="store_true", help="Also compute valid_ns -> train_all metrics.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: <split-root>.")
    parser.add_argument("--output-csv", default=None, help="Output CSV path. Overrides --out-dir.")
    parser.add_argument("--output-json", default=None, help="Output JSON path. Overrides --out-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    split_dirs = find_split_dirs(split_root)
    if not split_dirs:
        raise SystemExit(f"No dataset split folders found under: {split_root}")

    rows = [
        summarize_split_leakage(
            split_dir=split_dir,
            smiles_column=args.smiles_column,
            threshold=args.threshold,
            include_valid=args.include_valid,
        )
        for split_dir in split_dirs
    ]
    out_df = pd.DataFrame(rows).sort_values(["split_dir", "dataset"]).reset_index(drop=True)

    out_dir = Path(args.out_dir) if args.out_dir else split_root
    out_csv = Path(args.output_csv) if args.output_csv else out_dir / "leakage_metrics_summary.csv"
    out_json = Path(args.output_json) if args.output_json else out_dir / "leakage_metrics_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json.write_text(out_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved: {out_csv}")
    print(f"[OK] saved: {out_json}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
