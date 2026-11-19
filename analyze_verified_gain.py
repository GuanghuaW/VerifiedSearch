#!/usr/bin/env python3
"""Paired statistical analysis: VerifiedSearch vs Direct Qwen Multi."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["graph_name", "dataset_type", "diffusion", "budget"]

def holm_adjust(p_values):
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()

def matched_rank_biserial(differences):
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)

def bootstrap_ci(differences, rng, reps):
    n = len(differences)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = float(np.mean(differences[rng.integers(0, n, size=n)]))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)

def sign_flip_p(differences, rng, reps):
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 1.0
    observed = abs(float(np.mean(nonzero)))
    extreme = 0
    for _ in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(nonzero))
        if abs(float(np.mean(nonzero * signs))) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1.0) / (reps + 1.0)

def spread_col(df):
    for name in ["final_eval_spread", "final_spread", "mean_final_spread"]:
        if name in df.columns:
            return name
    raise ValueError(f"No spread column; columns={list(df.columns)}")

def prepare(path, label):
    df = pd.read_csv(path)
    missing = set(KEYS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    s = spread_col(df)
    out = df[KEYS + [s]].copy().rename(columns={s: label})
    out[label] = pd.to_numeric(out[label], errors="raise")
    return out

def summarize(frame, name, rng, bootstrap_reps, permutation_reps):
    direct = frame["direct_spread"].to_numpy(float)
    verified = frame["verified_spread"].to_numpy(float)
    diff = verified - direct
    low, high = bootstrap_ci(diff, rng, bootstrap_reps)
    p_sign = sign_flip_p(diff, rng, permutation_reps)
    nonzero = diff[diff != 0]
    if len(nonzero):
        test = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        p_w = float(test.pvalue)
        stat = float(test.statistic)
    else:
        p_w = 1.0
        stat = 0.0
    dm = float(np.mean(direct))
    vm = float(np.mean(verified))
    return {
        "group": name,
        "instances": len(frame),
        "direct_mean": dm,
        "verified_mean": vm,
        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "relative_improvement_percent": 100.0 * (vm - dm) / dm if dm else np.nan,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "wins_verified": int(np.sum(diff > 0)),
        "ties": int(np.sum(diff == 0)),
        "losses_verified": int(np.sum(diff < 0)),
        "rank_biserial": matched_rank_biserial(diff),
        "wilcoxon_statistic": stat,
        "p_wilcoxon_raw": p_w,
        "p_signflip_raw": p_sign,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--direct", type=Path, required=True)
    p.add_argument("--verified", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-reps", type=int, default=10000)
    p.add_argument("--permutation-reps", type=int, default=10000)
    p.add_argument("--base-seed", type=int, default=2026)
    a = p.parse_args()

    direct = prepare(a.direct, "direct_spread")
    verified = prepare(a.verified, "verified_spread")
    paired = direct.merge(verified, on=KEYS, how="inner", validate="one_to_one")
    rng = np.random.default_rng(a.base_seed)

    rows = [summarize(paired, "overall", rng, a.bootstrap_reps, a.permutation_reps)]
    for diffusion, group in paired.groupby("diffusion", sort=True):
        rows.append(summarize(group, f"diffusion:{diffusion}", rng, a.bootstrap_reps, a.permutation_reps))
    for budget, group in paired.groupby("budget", sort=True):
        rows.append(summarize(group, f"budget:{int(budget)}", rng, a.bootstrap_reps, a.permutation_reps))

    result = pd.DataFrame(rows)
    result["p_signflip_holm"] = holm_adjust(result["p_signflip_raw"].astype(float).tolist())
    result["p_wilcoxon_holm"] = holm_adjust(result["p_wilcoxon_raw"].astype(float).tolist())

    a.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output_dir / "verified_vs_direct_stats.csv", index=False)
    paired.to_csv(a.output_dir / "verified_vs_direct_paired_instances.csv", index=False)
    print(result.to_string(index=False))
    print(f"\nMatched instances: {len(paired)}")
    print(f"Saved to: {a.output_dir}")

if __name__ == "__main__":
    main()
