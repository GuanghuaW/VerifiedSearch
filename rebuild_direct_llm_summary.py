#!/usr/bin/env python3
"""Rebuild Direct-LLM MV and best-of-runs summaries from cached run JSON files.

No model is loaded and no generation is performed. The script reads existing
direct_llm_runs.json files, evaluates seed sets with the current RR caches, and
writes complete summary CSV files.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from baseline_common import (
    evaluate_seeds,
    iter_aggregate_paths,
    load_evaluation_context,
    load_json,
    make_summary_row,
    save_json,
    stable_seed,
    validate_seed_set,
    write_summary,
)
from rr_tools import build_node_rr_bitsets, estimate_spread, load_or_create_rr_sets


def majority_vote(rankings: Sequence[Sequence[int]], budget: int) -> list[int]:
    frequency: Counter[int] = Counter()
    ranks: dict[int, list[int]] = defaultdict(list)
    for ranking in rankings:
        for rank, node in enumerate(ranking[:budget], start=1):
            frequency[int(node)] += 1
            ranks[int(node)].append(rank)
    ordered = sorted(
        frequency,
        key=lambda node: (
            -frequency[node],
            statistics.fmean(ranks[node]),
            node,
        ),
    )
    if len(ordered) < budget:
        raise RuntimeError(
            f"Majority vote has only {len(ordered)} unique nodes for budget {budget}"
        )
    return ordered[:budget]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("VerifiedSearch/candidate_outputs_qwen35_9b"),
    )
    parser.add_argument(
        "--rr-cache-dir",
        type=Path,
        default=Path("VerifiedSearch/rr_cache_complete"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Exact model/mode directory containing dataset subdirectories.",
    )
    parser.add_argument("--method-key", required=True)
    parser.add_argument("--runs-label", type=int, default=10)
    parser.add_argument("--min-valid-runs", type=int, default=5)
    parser.add_argument("--tool-rr-sets", type=int, default=500)
    parser.add_argument("--eval-rr-sets", type=int, default=10000)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--overwrite-rr-cache", action="store_true")
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve()
    rr_cache_dir = args.rr_cache_dir.resolve()
    run_root = args.run_root.resolve()

    mv_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    aggregate_paths = list(iter_aggregate_paths(candidate_root))
    if not aggregate_paths:
        raise SystemExit("No aggregate files found.")

    for aggregate_path in aggregate_paths:
        aggregate, _, graph, eval_bitsets, _ = load_evaluation_context(
            aggregate_path,
            candidate_root=candidate_root,
            rr_cache_dir=rr_cache_dir,
            eval_rr_sets=args.eval_rr_sets,
            base_seed=args.base_seed,
            overwrite_rr_cache=args.overwrite_rr_cache,
        )
        graph_name = str(aggregate["graph_name"])
        dataset_type = str(aggregate["dataset_type"])
        diffusion = str(aggregate["diffusion"])

        instance_file = (
            run_root
            / dataset_type
            / graph_name
            / diffusion
            / "direct_llm_runs.json"
        )
        if not instance_file.exists():
            failures.append(
                {
                    "graph_name": graph_name,
                    "diffusion": diffusion,
                    "budget": None,
                    "error": "missing instance file",
                    "instance_file": str(instance_file),
                }
            )
            continue

        payload = load_json(instance_file)
        budgets = sorted(map(int, payload.get("runs_by_budget", {}).keys()))

        tool_seed = stable_seed(args.base_seed, graph_name, diffusion, "tool_rr")
        cache_dir = rr_cache_dir / dataset_type / graph_name / diffusion
        tool_cache = (
            cache_dir
            / f"tool_rr_{args.tool_rr_sets}_seed{tool_seed}.pkl.gz"
        )
        tool_rr = load_or_create_rr_sets(
            tool_cache,
            graph=graph,
            diffusion_name=diffusion,
            num_rr_sets=args.tool_rr_sets,
            seed=tool_seed,
            overwrite=args.overwrite_rr_cache,
        )
        tool_bitsets = build_node_rr_bitsets(tool_rr, graph.num_nodes)

        for budget in budgets:
            calls = list(payload["runs_by_budget"].get(str(budget), []))
            valid_calls = [
                call
                for call in calls
                if call.get("valid")
                and len(call.get("seed_nodes", [])) >= budget
            ]

            if len(valid_calls) < args.min_valid_runs:
                failures.append(
                    {
                        "graph_name": graph_name,
                        "diffusion": diffusion,
                        "budget": budget,
                        "valid_runs": len(valid_calls),
                        "required": args.min_valid_runs,
                        "instance_file": str(instance_file),
                    }
                )
                continue

            rankings = [
                validate_seed_set(
                    call["seed_nodes"],
                    budget=budget,
                    num_nodes=graph.num_nodes,
                )
                for call in valid_calls
            ]

            mv_seeds = majority_vote(rankings, budget)
            mv_metrics = evaluate_seeds(
                mv_seeds,
                graph=graph,
                eval_bitsets=eval_bitsets,
                eval_rr_sets=args.eval_rr_sets,
            )
            total_generation = sum(
                float(call.get("total_wall_seconds", 0.0))
                for call in valid_calls
            )
            fallback_runs = sum(
                bool(call.get("used_fallback"))
                for call in valid_calls
            )

            mv_rows.append(
                make_summary_row(
                    method=f"{args.method_key}_mv{args.runs_label}",
                    aggregate=aggregate,
                    graph=graph,
                    budget=budget,
                    seeds=mv_seeds,
                    metrics=mv_metrics,
                    eval_rr_sets=args.eval_rr_sets,
                    runtime_seconds=total_generation,
                    source_file=str(instance_file),
                    notes=(
                        f"Majority vote over {len(valid_calls)}/"
                        f"{args.runs_label} valid runs; "
                        f"fallback runs={fallback_runs}."
                    ),
                )
            )

            tool_scores: list[float] = []
            for call, seeds in zip(valid_calls, rankings):
                tool_estimate = estimate_spread(
                    seeds,
                    node_rr_bitsets=tool_bitsets,
                    num_nodes=graph.num_nodes,
                    num_rr_sets=args.tool_rr_sets,
                )
                eval_metrics = evaluate_seeds(
                    seeds,
                    graph=graph,
                    eval_bitsets=eval_bitsets,
                    eval_rr_sets=args.eval_rr_sets,
                )
                tool_scores.append(tool_estimate.spread)
                run_rows.append(
                    {
                        "method": f"{args.method_key}_run",
                        "graph_name": graph_name,
                        "dataset_type": dataset_type,
                        "diffusion": diffusion,
                        "budget": budget,
                        "run_index": call.get("run_index"),
                        "valid_runs": len(valid_calls),
                        "tool_spread": tool_estimate.spread,
                        "final_eval_spread": eval_metrics["spread"],
                        "normalized_final_spread": eval_metrics[
                            "normalized_spread"
                        ],
                        "runtime_seconds": call.get(
                            "total_wall_seconds",
                            0.0,
                        ),
                        "used_fallback": call.get(
                            "used_fallback",
                            False,
                        ),
                        "seed_set": json.dumps(seeds),
                        "source_file": str(instance_file),
                    }
                )

            best_index = max(
                range(len(valid_calls)),
                key=lambda index: (tool_scores[index], -index),
            )
            best_seeds = rankings[best_index]
            best_metrics = evaluate_seeds(
                best_seeds,
                graph=graph,
                eval_bitsets=eval_bitsets,
                eval_rr_sets=args.eval_rr_sets,
            )
            best_rows.append(
                make_summary_row(
                    method=f"{args.method_key}_best{args.runs_label}_tool",
                    aggregate=aggregate,
                    graph=graph,
                    budget=budget,
                    seeds=best_seeds,
                    metrics=best_metrics,
                    eval_rr_sets=args.eval_rr_sets,
                    runtime_seconds=total_generation,
                    source_file=str(instance_file),
                    notes=(
                        f"Best of {len(valid_calls)} valid runs selected "
                        f"using {args.tool_rr_sets} tool RR sets."
                    ),
                )
            )

    write_summary(
        mv_rows,
        run_root / f"summary_{args.method_key}_mv{args.runs_label}.csv",
    )
    write_summary(
        best_rows,
        run_root
        / f"summary_{args.method_key}_best{args.runs_label}_tool.csv",
    )

    if run_rows:
        run_path = run_root / f"runs_{args.method_key}.csv"
        with run_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(run_rows[0]),
            )
            writer.writeheader()
            writer.writerows(run_rows)

    index_path = run_root / "index_direct_llm_rebuilt.json"
    save_json(
        {
            "mv_rows": len(mv_rows),
            "best_rows": len(best_rows),
            "run_rows": len(run_rows),
            "failures": failures,
        },
        index_path,
    )

    print(f"MV rows: {len(mv_rows)}")
    print(f"Best rows: {len(best_rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Index: {index_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
