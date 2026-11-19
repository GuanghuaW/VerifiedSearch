#!/usr/bin/env python3
"""Merge updated baselines with VerifiedSearch results and create unified tables."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["graph_name", "dataset_type", "diffusion", "budget"]


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" not in df.columns:
        if "controller" in df.columns:
            df["method"] = df["controller"].astype(str)
        elif path.stem.startswith("summary_"):
            df["method"] = path.stem.removeprefix("summary_")
        else:
            raise ValueError(f"Cannot infer method from {path}")
    if "num_nodes" not in df.columns and "result_file" in df.columns:
        recovered_nodes = []
        for raw in df["result_file"].fillna(""):
            try:
                result_path = Path(str(raw))
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                recovered_nodes.append(int(payload["num_nodes"]))
            except Exception:
                recovered_nodes.append(np.nan)
        df["num_nodes"] = recovered_nodes
    if "normalized_final_spread" not in df.columns:
        if "num_nodes" in df.columns:
            df["normalized_final_spread"] = df["final_eval_spread"] / df["num_nodes"]
        else:
            df["normalized_final_spread"] = np.nan
    df["source_summary"] = str(path)
    required = {*KEYS, "method", "final_eval_spread"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def discover(paths: list[Path]) -> pd.DataFrame:
    frames = []
    seen_files: set[Path] = set()
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("summary_*.csv"))
        for path in files:
            path = path.resolve()
            if path in seen_files:
                continue
            seen_files.add(path)
            frames.append(load_summary(path))
    if not frames:
        raise RuntimeError("No summary CSV files found")
    data = pd.concat(frames, ignore_index=True, sort=False)
    duplicates = data.duplicated(KEYS + ["method"], keep=False)
    if duplicates.any():
        sample = data.loc[duplicates, KEYS + ["method", "source_summary"]].head(20)
        raise ValueError(
            "Duplicate method-instance rows detected. Pass only the intended summary roots:\n"
            + sample.to_string(index=False)
        )
    return data


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [math.nan] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_row(left: str, right: str, data: pd.DataFrame) -> dict[str, Any] | None:
    a = data[data["method"] == left][KEYS + ["final_eval_spread"]].rename(
        columns={"final_eval_spread": "left_value"}
    )
    b = data[data["method"] == right][KEYS + ["final_eval_spread"]].rename(
        columns={"final_eval_spread": "right_value"}
    )
    matched = a.merge(b, on=KEYS, how="inner")
    if matched.empty:
        return None
    diff = matched["left_value"].to_numpy() - matched["right_value"].to_numpy()
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        statistic, p_value = 0.0, 1.0
        rank_biserial = 0.0
    else:
        result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
        statistic, p_value = float(result.statistic), float(result.pvalue)
        abs_order = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
        pos = float(abs_order[nonzero > 0].sum())
        neg = float(abs_order[nonzero < 0].sum())
        rank_biserial = (pos - neg) / (pos + neg) if pos + neg else 0.0
    return {
        "left": left,
        "right": right,
        "instances": len(matched),
        "mean_difference_left_minus_right": float(diff.mean()),
        "median_difference_left_minus_right": float(np.median(diff)),
        "wins_left": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses_left": int((diff < 0).sum()),
        "wilcoxon_statistic": statistic,
        "p_raw": p_value,
        "rank_biserial": rank_biserial,
    }


def overall(data: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "instances": ("final_eval_spread", "size"),
        "mean_final_spread": ("final_eval_spread", "mean"),
        "median_final_spread": ("final_eval_spread", "median"),
        "mean_normalized_final_spread": ("normalized_final_spread", "mean"),
    }
    if "runtime_seconds" in data.columns:
        aggregations["mean_runtime_seconds"] = ("runtime_seconds", "mean")
    if "verified_swaps" in data.columns:
        aggregations["mean_verified_swaps"] = ("verified_swaps", "mean")
    return (
        data.groupby("method", as_index=False)
        .agg(**aggregations)
        .sort_values("mean_final_spread", ascending=False)
    )


def grouped(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        data.groupby(columns + ["method"], as_index=False)
        .agg(
            instances=("final_eval_spread", "size"),
            mean_final_spread=("final_eval_spread", "mean"),
            median_final_spread=("final_eval_spread", "median"),
            mean_normalized_final_spread=("normalized_final_spread", "mean"),
        )
        .sort_values(columns + ["mean_final_spread"], ascending=[True] * len(columns) + [False])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("VerifiedSearch/unified_baseline_analysis"),
    )
    parser.add_argument(
        "--reference-method",
        default="llm_fixed_randomfill",
        help="Pair every method against this main method.",
    )
    parser.add_argument("--all-pairs", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = discover(args.inputs)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    data.to_csv(output / "unified_instance_results.csv", index=False)
    overall_df = overall(data)
    overall_df.to_csv(output / "unified_overall.csv", index=False)
    grouped(data, ["diffusion"]).to_csv(output / "unified_by_diffusion.csv", index=False)
    grouped(data, ["dataset_type"]).to_csv(output / "unified_by_dataset_type.csv", index=False)
    grouped(data, ["budget"]).to_csv(output / "unified_by_budget.csv", index=False)
    grouped(data, ["dataset_type", "diffusion"]).to_csv(
        output / "unified_by_dataset_and_diffusion.csv", index=False
    )

    methods = sorted(data["method"].unique())
    if args.all_pairs:
        pairs = list(itertools.combinations(methods, 2))
    else:
        if args.reference_method not in methods:
            raise ValueError(
                f"Reference method {args.reference_method!r} not found. Available: {methods}"
            )
        pairs = [(args.reference_method, method) for method in methods if method != args.reference_method]

    rows = [row for left, right in pairs if (row := paired_row(left, right, data)) is not None]
    adjusted = holm_adjust([row["p_raw"] for row in rows]) if rows else []
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        pairwise = pairwise.sort_values("p_holm")
    pairwise.to_csv(output / "unified_pairwise.csv", index=False)

    print("\nOverall summary:\n")
    print(overall_df.to_string(index=False))
    print("\nPairwise comparisons:\n")
    print(pairwise.to_string(index=False) if not pairwise.empty else "No matched pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
