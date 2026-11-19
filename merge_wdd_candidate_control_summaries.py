#!/usr/bin/env python3
"""Merge budget-isolated WDD candidate-control summaries into 162-row source files.

Expected input layout:
  <root>/<source>/budget_<k>/summary_*.csv

Outputs:
  summary_wdd_qwen_proxy_clean.csv
  summary_wdd_out_degree_proxy_clean.csv
  summary_wdd_hybrid_proxy_clean.csv

The script validates that each source has exactly one row per
(graph_name, dataset_type, diffusion, budget).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["graph_name", "dataset_type", "diffusion", "budget"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--sources",
        nargs="+",
        default=["qwen", "out_degree", "hybrid"],
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        paths = sorted((args.input_root / source).glob("budget_*/summary_*.csv"))
        if not paths:
            raise RuntimeError(f"No summary CSVs for source={source}")

        frames = []
        for path in paths:
            df = pd.read_csv(path)
            missing = set(KEYS) - set(df.columns)
            if missing:
                raise RuntimeError(
                    f"{path} missing keys: {sorted(missing)}"
                )
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True, sort=False)
        duplicate_count = int(merged.duplicated(KEYS).sum())
        if duplicate_count:
            raise RuntimeError(
                f"{source}: {duplicate_count} duplicate matched instances"
            )

        merged = merged.sort_values(KEYS).reset_index(drop=True)
        if "method_label" in merged.columns:
            merged["method_label"] = f"wdd_{source}_proxy_clean"
        if "candidate_source" in merged.columns:
            merged["candidate_source"] = source

        out = (
            args.output_dir
            / f"summary_wdd_{source}_proxy_clean.csv"
        )
        merged.to_csv(out, index=False)

        print(
            f"{source:10s}: rows={len(merged):3d} "
            f"diffusions={merged['diffusion'].nunique()} "
            f"budgets={sorted(merged['budget'].unique().tolist())} "
            f"-> {out}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
