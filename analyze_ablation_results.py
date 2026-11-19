#!/usr/bin/env python3
"""Analyze matched IM ablation summaries with paired nonparametric tests."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["graph_name", "dataset_type", "diffusion", "budget"]


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def matched_rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    abs_values = np.abs(nonzero)
    order = pd.Series(abs_values).rank(method="average").to_numpy()
    positive = float(order[nonzero > 0].sum())
    negative = float(order[nonzero < 0].sum())
    denominator = positive + negative
    return 0.0 if denominator == 0 else (positive - negative) / denominator


def read_inputs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        required = set(KEYS + ["method_label", "final_eval_spread"])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["summary_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--inputs", type=Path, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = list(args.inputs or [])
    if args.root:
        paths.extend(sorted(args.root.rglob("summary_*.csv")))
    paths = sorted(set(path.resolve() for path in paths if path.exists()))
    if len(paths) < 2:
        raise RuntimeError("Provide at least two ablation summary CSV files.")

    data = read_inputs(paths)
    methods = sorted(data["method_label"].unique())
    key_sets = {
        method: set(map(tuple, data.loc[data.method_label == method, KEYS].to_numpy()))
        for method in methods
    }
    common = set.intersection(*(key_sets[method] for method in methods))
    if not common:
        raise RuntimeError("No common matched instances across methods.")
    common_frame = pd.DataFrame(sorted(common), columns=KEYS)
    matched = data.merge(common_frame, on=KEYS, how="inner")

    overall = (
        matched.groupby("method_label", as_index=False)
        .agg(
            instances=("final_eval_spread", "count"),
            candidate_source=("candidate_source", "first"),
            initialization=("initialization", "first"),
            search=("search", "first"),
            controller=("controller", "first"),
            candidate_pool_size=("candidate_pool_size", "mean"),
            mean_final_spread=("final_eval_spread", "mean"),
            median_final_spread=("final_eval_spread", "median"),
            mean_normalized_final_spread=("normalized_final_spread", "mean"),
            mean_gain_over_top=("gain_over_top", "mean"),
            mean_gain_over_initialization=("gain_over_initialization", "mean"),
            mean_verified_swaps=("verified_swaps", "mean"),
        )
        .sort_values("mean_final_spread", ascending=False)
    )

    pairwise_rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for left, right in itertools.combinations(methods, 2):
        left_frame = matched.loc[matched.method_label == left, KEYS + ["final_eval_spread"]]
        right_frame = matched.loc[matched.method_label == right, KEYS + ["final_eval_spread"]]
        paired = left_frame.merge(right_frame, on=KEYS, suffixes=("_left", "_right"))
        differences = (
            paired["final_eval_spread_left"].to_numpy(float)
            - paired["final_eval_spread_right"].to_numpy(float)
        )
        nonzero = differences[differences != 0]
        if nonzero.size:
            test = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
            statistic = float(test.statistic)
            p_raw = float(test.pvalue)
        else:
            statistic = 0.0
            p_raw = 1.0
        p_values.append(p_raw)
        pairwise_rows.append(
            {
                "left": left,
                "right": right,
                "instances": len(differences),
                "mean_difference_left_minus_right": float(np.mean(differences)),
                "median_difference_left_minus_right": float(np.median(differences)),
                "wins_left": int(np.sum(differences > 0)),
                "ties": int(np.sum(differences == 0)),
                "losses_left": int(np.sum(differences < 0)),
                "wilcoxon_statistic": statistic,
                "p_raw": p_raw,
                "rank_biserial": matched_rank_biserial(differences),
            }
        )

    adjusted = holm_adjust(p_values)
    for row, p_holm in zip(pairwise_rows, adjusted):
        row["p_holm"] = p_holm
    pairwise = pd.DataFrame(pairwise_rows).sort_values("p_holm")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_path = args.output_dir / "ablation_overall.csv"
    pairwise_path = args.output_dir / "ablation_pairwise.csv"
    overall.to_csv(overall_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)

    print(f"Common matched instances: {len(common)}")
    print("\nOverall summary:\n")
    print(overall.to_string(index=False))
    print("\nPaired comparisons:\n")
    print(pairwise.to_string(index=False))
    print(f"\nSaved: {overall_path}")
    print(f"Saved: {pairwise_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
