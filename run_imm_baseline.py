#!/usr/bin/env python3
"""Run a reproducible IMM baseline on the current VerifiedSearch graphs.

This implementation follows the IMM sampling equations from Tang, Shi, and
Xiao (SIGMOD 2015), with the "regenerate final RR sets" workaround described
by Wei Chen (2018) enabled by default. Seed selection and final evaluation use
disjoint RR-set samples.

The script expects baseline_common.py and rr_tools.py in the same directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

from baseline_common import (
    evaluate_seeds,
    iter_aggregate_paths,
    load_evaluation_context,
    load_json,
    make_summary_row,
    save_json,
    stable_seed,
    write_summary,
)
from rr_tools import (
    GraphView,
    build_node_rr_bitsets,
    parse_diffusion_name,
    sample_one_rr_set,
)


def log_binomial(n: int, k: int) -> float:
    if not 0 <= k <= n:
        raise ValueError(f"Require 0 <= k <= n; got n={n}, k={k}.")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def greedy_rr_max_cover(
    rr_sets: Sequence[Sequence[int]],
    *,
    num_nodes: int,
    budget: int,
) -> tuple[list[int], float, int]:
    if not rr_sets:
        raise ValueError("RR-set collection is empty.")
    if budget <= 0 or budget > num_nodes:
        raise ValueError(
            f"Budget must be in 1..{num_nodes}; got {budget}."
        )

    bitsets = build_node_rr_bitsets(rr_sets, num_nodes)
    covered = 0
    selected: list[int] = []
    selected_set: set[int] = set()

    for _ in range(budget):
        best_node = None
        best_gain = -1

        for node in range(1, num_nodes + 1):
            if node in selected_set:
                continue
            gain = (bitsets[node] & ~covered).bit_count()
            if gain > best_gain or (
                gain == best_gain
                and (best_node is None or node < best_node)
            ):
                best_gain = gain
                best_node = node

        if best_node is None:
            raise RuntimeError("Could not choose the next seed node.")

        selected.append(best_node)
        selected_set.add(best_node)
        covered |= bitsets[best_node]

    covered_count = covered.bit_count()
    coverage_fraction = covered_count / len(rr_sets)
    return selected, coverage_fraction, covered_count


def append_rr_sets(
    rr_sets: list[tuple[int, ...]],
    *,
    target_size: int,
    graph: GraphView,
    diffusion_name: str,
    rng: random.Random,
    max_rr_sets: int,
) -> None:
    if target_size > max_rr_sets:
        raise RuntimeError(
            f"IMM requested {target_size:,} RR sets, exceeding "
            f"--max-rr-sets={max_rr_sets:,}. Increase the cap deliberately "
            "or use a larger epsilon."
        )

    diffusion_kind, uniform_p = parse_diffusion_name(diffusion_name)
    while len(rr_sets) < target_size:
        rr_sets.append(
            sample_one_rr_set(
                graph,
                rng,
                diffusion_kind,
                uniform_p,
            )
        )


def imm_select(
    *,
    graph: GraphView,
    diffusion_name: str,
    budget: int,
    epsilon: float,
    ell: float,
    seed: int,
    final_rr_policy: str,
    max_rr_sets: int,
) -> tuple[list[int], dict[str, Any]]:
    if not 0.0 < epsilon < 1.0:
        raise ValueError("--epsilon must lie in (0, 1).")
    if graph.num_nodes <= 1:
        raise ValueError("IMM requires a graph with at least two nodes.")

    n = graph.num_nodes
    log_n = math.log(n)
    log_2_n = math.log2(n)
    log_choose = log_binomial(n, budget)

    # IMM Algorithm 1 adjusts ell before invoking Sampling.
    ell_adjusted = ell + math.log(2.0) / log_n
    epsilon_prime = math.sqrt(2.0) * epsilon

    lambda_prime = (
        (2.0 + (2.0 / 3.0) * epsilon_prime)
        * (
            log_choose
            + ell_adjusted * log_n
            + math.log(max(log_2_n, 1.0))
        )
        * n
        / (epsilon_prime**2)
    )

    rng_lb = random.Random(
        stable_seed(seed, "imm", "lower_bound")
    )
    lower_rr_sets: list[tuple[int, ...]] = []
    lower_bound = 1.0
    lower_bound_round = None
    lower_bound_coverage = None
    lower_bound_seed_set: list[int] | None = None
    lower_bound_trace: list[dict[str, Any]] = []

    max_round = max(1, int(math.floor(log_2_n)) - 1)

    for round_index in range(1, max_round + 1):
        x = n / (2.0**round_index)
        theta_i = lambda_prime / x
        target = max(1, int(math.ceil(theta_i)))

        append_rr_sets(
            lower_rr_sets,
            target_size=target,
            graph=graph,
            diffusion_name=diffusion_name,
            rng=rng_lb,
            max_rr_sets=max_rr_sets,
        )

        seeds_i, fraction_i, covered_i = greedy_rr_max_cover(
            lower_rr_sets,
            num_nodes=n,
            budget=budget,
        )
        estimated_spread_i = n * fraction_i

        lower_bound_trace.append(
            {
                "round": round_index,
                "x": x,
                "theta_i": theta_i,
                "rr_sets": len(lower_rr_sets),
                "coverage_fraction": fraction_i,
                "covered_rr_sets": covered_i,
                "estimated_spread": estimated_spread_i,
            }
        )

        if estimated_spread_i >= (1.0 + epsilon_prime) * x:
            lower_bound = estimated_spread_i / (
                1.0 + epsilon_prime
            )
            lower_bound_round = round_index
            lower_bound_coverage = fraction_i
            lower_bound_seed_set = seeds_i
            break

    alpha = math.sqrt(
        ell_adjusted * log_n + math.log(2.0)
    )
    beta = math.sqrt(
        (1.0 - 1.0 / math.e)
        * (
            log_choose
            + ell_adjusted * log_n
            + math.log(2.0)
        )
    )
    lambda_star = (
        2.0
        * n
        * (
            (1.0 - 1.0 / math.e) * alpha
            + beta
        ) ** 2
        / (epsilon**2)
    )
    theta = lambda_star / max(lower_bound, 1e-12)
    final_target = max(1, int(math.ceil(theta)))

    if final_rr_policy == "regenerate":
        # Workaround 1 from Chen (2018): final RR sets are independent
        # of the RR sets used to estimate the stopping-time lower bound.
        final_rr_sets: list[tuple[int, ...]] = []
        rng_final = random.Random(
            stable_seed(seed, "imm", "final")
        )
    elif final_rr_policy == "reuse":
        # Original IMM behavior: reuse and extend the lower-bound sample.
        final_rr_sets = list(lower_rr_sets)
        rng_final = rng_lb
    else:
        raise ValueError(
            f"Unknown final RR policy: {final_rr_policy}"
        )

    append_rr_sets(
        final_rr_sets,
        target_size=final_target,
        graph=graph,
        diffusion_name=diffusion_name,
        rng=rng_final,
        max_rr_sets=max_rr_sets,
    )

    seeds, final_fraction, final_covered = greedy_rr_max_cover(
        final_rr_sets,
        num_nodes=n,
        budget=budget,
    )

    metadata: dict[str, Any] = {
        "algorithm": (
            "IMM-W1"
            if final_rr_policy == "regenerate"
            else "IMM-original"
        ),
        "epsilon": epsilon,
        "ell": ell,
        "ell_adjusted": ell_adjusted,
        "epsilon_prime": epsilon_prime,
        "lambda_prime": lambda_prime,
        "lambda_star": lambda_star,
        "lower_bound": lower_bound,
        "lower_bound_round": lower_bound_round,
        "lower_bound_coverage": lower_bound_coverage,
        "lower_bound_seed_set": lower_bound_seed_set,
        "lower_bound_rr_sets": len(lower_rr_sets),
        "lower_bound_trace": lower_bound_trace,
        "theta_final": theta,
        "final_rr_sets": len(final_rr_sets),
        "final_covered_rr_sets": final_covered,
        "final_rr_coverage_fraction": final_fraction,
        "final_rr_estimated_spread": n * final_fraction,
        "final_rr_policy": final_rr_policy,
        "seed": seed,
    }
    return seeds, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(
            "VerifiedSearch/candidate_outputs_qwen35_9b"
        ),
    )
    parser.add_argument(
        "--rr-cache-dir",
        type=Path,
        default=Path("VerifiedSearch/rr_cache_complete"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "VerifiedSearch/baseline_outputs_v2/imm"
        ),
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[10, 20, 30],
    )
    parser.add_argument(
        "--graphs",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--dataset-types",
        nargs="*",
        choices=["synthetic", "real"],
        default=None,
    )
    parser.add_argument(
        "--diffusions",
        nargs="*",
        default=None,
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--ell", type=float, default=1.0)
    parser.add_argument(
        "--final-rr-policy",
        choices=["regenerate", "reuse"],
        default="regenerate",
        help=(
            "regenerate = corrected IMM workaround 1; "
            "reuse = original IMM sample reuse."
        ),
    )
    parser.add_argument(
        "--eval-rr-sets",
        type=int,
        default=10000,
    )
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument(
        "--max-rr-sets",
        type=int,
        default=5_000_000,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-rr-cache",
        action="store_true",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    candidate_root = args.candidate_root.resolve()
    rr_cache_dir = args.rr_cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_names = set(args.graphs) if args.graphs else None
    dataset_types = (
        set(args.dataset_types)
        if args.dataset_types
        else None
    )
    diffusions = (
        set(args.diffusions)
        if args.diffusions
        else None
    )

    method = (
        "imm_w1"
        if args.final_rr_policy == "regenerate"
        else "imm_original"
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    aggregate_paths = list(
        iter_aggregate_paths(
            candidate_root,
            graph_names=graph_names,
            dataset_types=dataset_types,
            diffusions=diffusions,
        )
    )
    total_jobs = len(aggregate_paths) * len(args.budgets)
    job_index = 0

    for aggregate_path in aggregate_paths:
        try:
            (
                aggregate,
                graph_path,
                graph,
                eval_bitsets,
                eval_seed,
            ) = load_evaluation_context(
                aggregate_path,
                candidate_root=candidate_root,
                rr_cache_dir=rr_cache_dir,
                eval_rr_sets=args.eval_rr_sets,
                base_seed=args.base_seed,
                overwrite_rr_cache=args.overwrite_rr_cache,
            )
        except Exception as exc:
            failures.append(
                {
                    "aggregate_file": str(aggregate_path),
                    "error": repr(exc),
                }
            )
            continue

        graph_name = str(aggregate["graph_name"])
        dataset_type = str(aggregate["dataset_type"])
        diffusion = str(aggregate["diffusion"])

        for budget in args.budgets:
            job_index += 1
            print(
                f"[{job_index}/{total_jobs}] "
                f"{graph_name} | {diffusion} | k={budget}",
                flush=True,
            )

            result_file = (
                output_dir
                / "results"
                / dataset_type
                / graph_name
                / diffusion
                / f"k{budget}.json"
            )

            try:
                if result_file.exists() and not args.overwrite:
                    result = load_json(result_file)
                    seeds = list(map(int, result["seeds"]))
                    runtime_seconds = float(
                        result["runtime_seconds"]
                    )
                    metadata = result["imm_metadata"]
                else:
                    instance_seed = stable_seed(
                        args.base_seed,
                        "imm",
                        graph_name,
                        diffusion,
                        budget,
                        args.epsilon,
                        args.ell,
                        args.final_rr_policy,
                    )
                    started = time.perf_counter()
                    seeds, metadata = imm_select(
                        graph=graph,
                        diffusion_name=diffusion,
                        budget=budget,
                        epsilon=args.epsilon,
                        ell=args.ell,
                        seed=instance_seed,
                        final_rr_policy=args.final_rr_policy,
                        max_rr_sets=args.max_rr_sets,
                    )
                    runtime_seconds = (
                        time.perf_counter() - started
                    )

                    result = {
                        "method": method,
                        "graph_name": graph_name,
                        "dataset_type": dataset_type,
                        "diffusion": diffusion,
                        "budget": budget,
                        "graph_file": str(graph_path),
                        "aggregate_file": str(aggregate_path),
                        "seeds": seeds,
                        "runtime_seconds": runtime_seconds,
                        "imm_metadata": metadata,
                    }
                    save_json(result, result_file)

                metrics = evaluate_seeds(
                    seeds,
                    graph=graph,
                    eval_bitsets=eval_bitsets,
                    eval_rr_sets=args.eval_rr_sets,
                )
                rows.append(
                    make_summary_row(
                        method=method,
                        aggregate=aggregate,
                        graph=graph,
                        budget=budget,
                        seeds=seeds,
                        metrics=metrics,
                        eval_rr_sets=args.eval_rr_sets,
                        runtime_seconds=runtime_seconds,
                        source_file=str(result_file),
                        notes=(
                            f"epsilon={args.epsilon}; "
                            f"ell={args.ell}; "
                            f"policy={args.final_rr_policy}; "
                            f"selection_rr_sets="
                            f"{metadata['final_rr_sets']}; "
                            f"independent_eval_seed={eval_seed}"
                        ),
                    )
                )
            except Exception as exc:
                failure = {
                    "graph_name": graph_name,
                    "dataset_type": dataset_type,
                    "diffusion": diffusion,
                    "budget": budget,
                    "error": repr(exc),
                    "result_file": str(result_file),
                }
                failures.append(failure)
                print(f"ERROR: {failure}", flush=True)

    summary_path = output_dir / f"summary_{method}.csv"
    write_summary(rows, summary_path)
    save_json(
        {
            "method": method,
            "rows": len(rows),
            "expected_jobs": total_jobs,
            "failures": failures,
            "parameters": {
                "epsilon": args.epsilon,
                "ell": args.ell,
                "final_rr_policy": args.final_rr_policy,
                "eval_rr_sets": args.eval_rr_sets,
                "base_seed": args.base_seed,
                "max_rr_sets": args.max_rr_sets,
            },
        },
        output_dir / f"index_{method}.json",
    )

    print(f"Summary rows: {len(rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Summary: {summary_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
