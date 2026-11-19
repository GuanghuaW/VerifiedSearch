#!/usr/bin/env python3
"""Compare fixed-budget verified-search controllers on matched instances."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

KEYS = ["graph_name", "dataset_type", "diffusion", "budget"]


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    positive = float(ranks[diff > 0].sum())
    negative = float(ranks[diff < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def read_controller_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(KEYS + ["controller", "final_eval_spread"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        duplicated = frame.loc[frame.duplicated(KEYS, keep=False), KEYS]
        raise ValueError(f"Duplicate instances in {path}:\n{duplicated.head()}")
    return frame


def attach_num_nodes(frame: pd.DataFrame) -> pd.DataFrame:
    cache: dict[str, int] = {}
    values: list[int | float] = []
    for raw_path in frame["result_file"]:
        path = str(raw_path)
        if path not in cache:
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                cache[path] = int(payload["num_nodes"])
            except Exception:
                cache[path] = 0
        values.append(cache[path] if cache[path] > 0 else np.nan)
    result = frame.copy()
    result["num_nodes"] = values
    result["normalized_final_spread"] = result["final_eval_spread"] / result["num_nodes"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("VerifiedSearch/verified_search_fixed_budget_outputs"),
    )
    parser.add_argument("--output-prefix", default="fixed_budget_comparison")
    args = parser.parse_args()

    paths = sorted(args.root.glob("summary_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No summary_*.csv files under {args.root}")

    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = attach_num_nodes(read_controller_csv(path))
        controller_values = frame["controller"].dropna().unique()
        if len(controller_values) != 1:
            raise ValueError(f"Expected one controller in {path}: {controller_values}")
        controller = str(controller_values[0])
        frames[controller] = frame
        print(f"Loaded {controller}: {len(frame)} rows from {path}")

    common_keys: set[tuple] | None = None
    for frame in frames.values():
        keys = set(map(tuple, frame[KEYS].to_numpy()))
        common_keys = keys if common_keys is None else common_keys & keys
    assert common_keys is not None
    print(f"Common matched instances: {len(common_keys)}")

    filtered: dict[str, pd.DataFrame] = {}
    common_index = pd.MultiIndex.from_tuples(sorted(common_keys), names=KEYS)
    for controller, frame in frames.items():
        indexed = frame.set_index(KEYS).loc[common_index].sort_index()
        filtered[controller] = indexed

    overall_rows = []
    exhaustive_name = next(
        (name for name in filtered if name.startswith("exhaustive_fixed")), None
    )
    exhaustive = filtered.get(exhaustive_name) if exhaustive_name else None

    for controller, frame in filtered.items():
        row = {
            "controller": controller,
            "instances": len(frame),
            "mean_final_spread": frame["final_eval_spread"].mean(),
            "mean_normalized_final_spread": frame["normalized_final_spread"].mean(),
            "mean_gain_consensus": frame["gain_over_consensus"].mean(),
            "mean_gain_initial": frame["gain_over_initialization"].mean(),
            "median_gain_initial": frame["gain_over_initialization"].median(),
            "mean_accepted_swaps": frame["accepted_swaps"].mean(),
            "mean_verified_swaps": frame["verified_swaps"].mean(),
            "mean_budget_utilization": frame["budget_utilization"].mean(),
        }
        if "llm_selected_proposals" in frame.columns:
            row["mean_llm_selected_proposals"] = frame[
                "llm_selected_proposals"
            ].mean()
            row["mean_fill_selected_proposals"] = frame[
                "fill_selected_proposals"
            ].mean()
        if exhaustive is not None and controller != exhaustive_name:
            row["mean_regret_to_exhaustive"] = (
                exhaustive["final_eval_spread"] - frame["final_eval_spread"]
            ).mean()
            denominator = exhaustive["gain_over_initialization"].to_numpy(float)
            numerator = frame["gain_over_initialization"].to_numpy(float)
            valid = denominator > 1e-12
            row["mean_gain_recovery_vs_exhaustive"] = (
                float(np.mean(numerator[valid] / denominator[valid]))
                if valid.any()
                else np.nan
            )
        overall_rows.append(row)

    overall = pd.DataFrame(overall_rows).sort_values(
        "mean_final_spread", ascending=False
    )
    overall_path = args.root / f"{args.output_prefix}_overall.csv"
    overall.to_csv(overall_path, index=False)
    print("\nOverall summary:\n")
    print(overall.to_string(index=False))

    pairwise_rows = []
    raw_p_values = []
    pair_names = []
    for left, right in itertools.combinations(sorted(filtered), 2):
        x = filtered[left]["final_eval_spread"].to_numpy(float)
        y = filtered[right]["final_eval_spread"].to_numpy(float)
        diff = x - y
        nonzero = diff[diff != 0]
        if nonzero.size:
            test = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        else:
            statistic = 0.0
            p_value = 1.0
        pair_names.append((left, right))
        raw_p_values.append(p_value)
        pairwise_rows.append(
            {
                "left": left,
                "right": right,
                "instances": len(x),
                "mean_difference_left_minus_right": float(diff.mean()),
                "median_difference_left_minus_right": float(np.median(diff)),
                "wins_left": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "losses_left": int((diff < 0).sum()),
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "rank_biserial": rank_biserial(x, y),
            }
        )

    adjusted = holm_adjust(raw_p_values)
    for row, p_adjusted in zip(pairwise_rows, adjusted):
        row["p_holm"] = p_adjusted
    pairwise = pd.DataFrame(pairwise_rows).sort_values(["p_holm", "p_raw"])
    pairwise_path = args.root / f"{args.output_prefix}_pairwise.csv"
    pairwise.to_csv(pairwise_path, index=False)
    print("\nPaired comparisons:\n")
    print(pairwise.to_string(index=False))
    print(f"\nSaved: {overall_path}")
    print(f"Saved: {pairwise_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
