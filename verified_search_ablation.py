#!/usr/bin/env python3
"""Run candidate-source and component ablations for RR-verified IM search.

This is a thin experimental layer over verified_search_fixed_budget.py. It adds:

* --initialization {top,diversity}
* --search {none,fixed}
* --method-label for collision-free output/summary names

RR-set generation, fixed-budget repair, controllers, and evaluation are reused
from the validated fixed-budget implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import verified_search_fixed_budget as base  # noqa: E402
from rr_tools import (  # noqa: E402
    build_node_rr_bitsets,
    load_or_create_rr_sets,
    load_processed_graph,
)

LOGGER = logging.getLogger("verified_search_ablation")


def sanitize_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    label = label.strip("_.-")
    if not label:
        raise ValueError("--method-label must contain at least one safe character.")
    return label


def no_search_diagnostics() -> dict[str, Any]:
    return {
        "verification_budget": 0,
        "verified_swaps": 0,
        "budget_utilization": 0.0,
        "accepted_swaps": 0,
        "proposal_batches": 0,
        "llm_selected_proposals": 0,
        "fill_selected_proposals": 0,
        "stop_reason": "search_disabled",
    }


def process_aggregate(
    aggregate_path: Path,
    *,
    args: argparse.Namespace,
    controller: base.SwapController | None,
) -> tuple[dict[str, Any], Path]:
    aggregate = base.load_json(aggregate_path)
    graph_name = str(aggregate["graph_name"])
    dataset_type = str(aggregate["dataset_type"])
    diffusion = str(aggregate["diffusion"])
    candidate_source = str(
        aggregate.get("ablation", {}).get("candidate_source", "qwen_multi_original")
    )

    graph_path = base.resolve_graph_path(str(aggregate["graph_file"]), args.candidate_root)
    _, graph = load_processed_graph(graph_path)

    relative_parent = aggregate_path.parent.relative_to(args.candidate_root)
    output_path = args.output_dir / relative_parent / f"verified_{args.method_label}.json"
    if output_path.exists() and not args.overwrite:
        LOGGER.info("Skipping existing result: %s", output_path)
        return base.load_json(output_path), output_path

    cache_dir = args.rr_cache_dir / dataset_type / graph_name / diffusion
    tool_seed = base.stable_seed(args.base_seed, graph_name, diffusion, "tool_rr")
    eval_seed = base.stable_seed(args.base_seed, graph_name, diffusion, "eval_rr")
    tool_cache = cache_dir / f"tool_rr_{args.tool_rr_sets}_seed{tool_seed}.pkl.gz"
    eval_cache = cache_dir / f"eval_rr_{args.eval_rr_sets}_seed{eval_seed}.pkl.gz"

    tool_rr = load_or_create_rr_sets(
        tool_cache,
        graph=graph,
        diffusion_name=diffusion,
        num_rr_sets=args.tool_rr_sets,
        seed=tool_seed,
        overwrite=args.overwrite_rr_cache,
    )
    eval_rr = load_or_create_rr_sets(
        eval_cache,
        graph=graph,
        diffusion_name=diffusion,
        num_rr_sets=args.eval_rr_sets,
        seed=eval_seed,
        overwrite=args.overwrite_rr_cache,
    )
    tool_bitsets = build_node_rr_bitsets(tool_rr, graph.num_nodes)
    eval_bitsets = build_node_rr_bitsets(eval_rr, graph.num_nodes)

    candidate_pool = [int(value) for value in aggregate.get("candidate_pool_ids", [])]
    scores = base.candidate_score_map(aggregate)
    if not candidate_pool:
        raise ValueError(f"No candidate_pool_ids in {aggregate_path}")

    results: dict[str, Any] = {}
    for budget in map(int, aggregate.get("budgets", [])):
        if budget > len(candidate_pool):
            LOGGER.warning(
                "Skipping graph=%s budget=%d: pool size=%d",
                graph_name,
                budget,
                len(candidate_pool),
            )
            continue

        top_seed = aggregate.get("consensus_seed_baselines", {}).get(str(budget))
        if not top_seed:
            top_seed = candidate_pool[:budget]
        top_seed = [int(value) for value in top_seed]

        if args.initialization == "top":
            initial = top_seed.copy()
            initialization_trace: list[dict[str, Any]] = []
        else:
            initial, initialization_trace = base.build_consensus_diversity_initialization(
                candidate_pool=candidate_pool,
                consensus_scores=scores,
                budget=budget,
                bitsets=tool_bitsets,
            )

        if args.search == "none":
            final = initial.copy()
            history: list[dict[str, Any]] = []
            diagnostics = no_search_diagnostics()
        else:
            if controller is None:
                raise RuntimeError("Search is enabled but no controller was built.")
            final, history, diagnostics = base.run_verified_repair(
                initial_seed_set=initial,
                candidate_pool=candidate_pool,
                consensus_scores=scores,
                graph=graph,
                bitsets=tool_bitsets,
                num_rr_sets=args.tool_rr_sets,
                controller=controller,
                verification_budget=args.verification_budget,
                batch_size=args.batch_size,
                max_accepted_swaps=args.max_accepted_swaps,
                num_weak=args.weak_seeds,
                num_additions=args.addition_candidates,
                min_rr_gain=args.min_rr_gain,
                base_seed=base.stable_seed(args.base_seed, graph_name, diffusion, budget),
                llm_fill_strategy=args.llm_fill,
            )

        top_metrics = base.method_metrics(
            top_seed,
            tool_bitsets=tool_bitsets,
            eval_bitsets=eval_bitsets,
            graph=graph,
            tool_rr_count=args.tool_rr_sets,
            eval_rr_count=args.eval_rr_sets,
        )
        initial_metrics = base.method_metrics(
            initial,
            tool_bitsets=tool_bitsets,
            eval_bitsets=eval_bitsets,
            graph=graph,
            tool_rr_count=args.tool_rr_sets,
            eval_rr_count=args.eval_rr_sets,
        )
        final_metrics = base.method_metrics(
            final,
            tool_bitsets=tool_bitsets,
            eval_bitsets=eval_bitsets,
            graph=graph,
            tool_rr_count=args.tool_rr_sets,
            eval_rr_count=args.eval_rr_sets,
        )

        top_eval = float(top_metrics["evaluation_estimate"]["spread"])
        initial_eval = float(initial_metrics["evaluation_estimate"]["spread"])
        final_eval = float(final_metrics["evaluation_estimate"]["spread"])
        results[str(budget)] = {
            "top_ranking_baseline": top_metrics,
            "initialization": initial_metrics,
            "verified_search": final_metrics,
            "initialization_trace": initialization_trace,
            "repair_history": history,
            "repair_diagnostics": diagnostics,
            "evaluation_gain_over_top": final_eval - top_eval,
            "evaluation_gain_over_initialization": final_eval - initial_eval,
            "num_accepted_swaps": int(diagnostics["accepted_swaps"]),
            "num_verified_swaps": int(diagnostics["verified_swaps"]),
            "verification_budget_utilization": float(diagnostics["budget_utilization"]),
        }

    payload = {
        "version": 1,
        "method_label": args.method_label,
        "graph_name": graph_name,
        "dataset_type": dataset_type,
        "graph_file": str(graph_path),
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "diffusion": diffusion,
        "candidate_source": candidate_source,
        "candidate_aggregate_file": str(aggregate_path),
        "candidate_pool_size": len(candidate_pool),
        "initialization": args.initialization,
        "search": args.search,
        "controller": "none" if controller is None else controller.name,
        "configuration": {
            "tool_rr_sets": args.tool_rr_sets,
            "eval_rr_sets": args.eval_rr_sets,
            "verification_budget": args.verification_budget if args.search == "fixed" else 0,
            "batch_size": args.batch_size,
            "max_accepted_swaps": args.max_accepted_swaps,
            "weak_seeds": args.weak_seeds,
            "addition_candidates": args.addition_candidates,
            "base_seed": args.base_seed,
        },
        "budgets": results,
    }
    base.save_json(payload, output_path)
    return payload, output_path


def write_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method_label",
        "candidate_source",
        "initialization",
        "search",
        "controller",
        "graph_name",
        "dataset_type",
        "diffusion",
        "budget",
        "num_nodes",
        "candidate_pool_size",
        "top_eval_spread",
        "initial_eval_spread",
        "final_eval_spread",
        "normalized_final_spread",
        "gain_over_top",
        "gain_over_initialization",
        "accepted_swaps",
        "verified_swaps",
        "verification_budget",
        "budget_utilization",
        "llm_selected_proposals",
        "fill_selected_proposals",
        "stop_reason",
        "result_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IM component/candidate ablations.")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rr-cache-dir", type=Path, default=SCRIPT_DIR / "rr_cache_complete"
    )
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--initialization", choices=("top", "diversity"), default="diversity")
    parser.add_argument("--search", choices=("none", "fixed"), default="fixed")
    parser.add_argument(
        "--controller",
        choices=("proxy_fixed", "random_fixed", "exhaustive_fixed", "llm_fixed"),
        default="proxy_fixed",
    )
    parser.add_argument("--tool-rr-sets", type=int, default=500)
    parser.add_argument("--eval-rr-sets", type=int, default=10000)
    parser.add_argument("--verification-budget", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-accepted-swaps", type=int, default=3)
    parser.add_argument("--weak-seeds", type=int, default=3)
    parser.add_argument("--addition-candidates", type=int, default=10)
    parser.add_argument("--min-rr-gain", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--graphs", nargs="*", default=None)
    parser.add_argument("--dataset-types", nargs="*", choices=("synthetic", "real"), default=None)
    parser.add_argument("--diffusions", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-rr-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--llm-proposal-retries", type=int, default=2)
    parser.add_argument("--llm-fill", choices=("random", "proxy", "none"), default="random")
    parser.add_argument("--trust-remote-code", action="store_true")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true")
    thinking.add_argument("--no-enable-thinking", action="store_false", dest="enable_thinking")
    parser.set_defaults(enable_thinking=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.method_label = sanitize_label(args.method_label)
    args.candidate_root = args.candidate_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.rr_cache_dir = args.rr_cache_dir.expanduser().resolve()

    paths = base.discover_aggregates(args.candidate_root)
    selected: list[Path] = []
    graph_filter = set(args.graphs or [])
    type_filter = set(args.dataset_types or [])
    diffusion_filter = set(args.diffusions or [])
    for path in paths:
        aggregate = base.load_json(path)
        if graph_filter and str(aggregate.get("graph_name")) not in graph_filter:
            continue
        if type_filter and str(aggregate.get("dataset_type")) not in type_filter:
            continue
        if diffusion_filter and str(aggregate.get("diffusion")) not in diffusion_filter:
            continue
        selected.append(path)

    if not selected:
        LOGGER.error("No aggregate.json files matched under %s", args.candidate_root)
        return 1
    LOGGER.info("Selected %d graph/diffusion jobs for %s", len(selected), args.method_label)
    if args.dry_run:
        for path in selected:
            print(path)
        return 0

    controller = None if args.search == "none" else base.build_controller(args)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    for aggregate_path in selected:
        try:
            payload, output_path = process_aggregate(
                aggregate_path,
                args=args,
                controller=controller,
            )
            for budget, item in payload["budgets"].items():
                diagnostics = item["repair_diagnostics"]
                final_spread = float(item["verified_search"]["evaluation_estimate"]["spread"])
                rows.append(
                    {
                        "method_label": payload["method_label"],
                        "candidate_source": payload["candidate_source"],
                        "initialization": payload["initialization"],
                        "search": payload["search"],
                        "controller": payload["controller"],
                        "graph_name": payload["graph_name"],
                        "dataset_type": payload["dataset_type"],
                        "diffusion": payload["diffusion"],
                        "budget": int(budget),
                        "num_nodes": payload["num_nodes"],
                        "candidate_pool_size": payload["candidate_pool_size"],
                        "top_eval_spread": item["top_ranking_baseline"]["evaluation_estimate"]["spread"],
                        "initial_eval_spread": item["initialization"]["evaluation_estimate"]["spread"],
                        "final_eval_spread": final_spread,
                        "normalized_final_spread": final_spread / payload["num_nodes"],
                        "gain_over_top": item["evaluation_gain_over_top"],
                        "gain_over_initialization": item["evaluation_gain_over_initialization"],
                        "accepted_swaps": item["num_accepted_swaps"],
                        "verified_swaps": item["num_verified_swaps"],
                        "verification_budget": payload["configuration"]["verification_budget"],
                        "budget_utilization": item["verification_budget_utilization"],
                        "llm_selected_proposals": diagnostics.get("llm_selected_proposals", 0),
                        "fill_selected_proposals": diagnostics.get("fill_selected_proposals", 0),
                        "stop_reason": diagnostics.get("stop_reason", "unknown"),
                        "result_file": str(output_path),
                    }
                )
            LOGGER.info("Completed %s %s", payload["graph_name"], payload["diffusion"])
        except Exception as exc:
            LOGGER.exception("Failed aggregate=%s", aggregate_path)
            failures.append({"aggregate": str(aggregate_path), "error": str(exc)})

    summary_path = args.output_dir / f"summary_{args.method_label}.csv"
    write_summary(rows, summary_path)
    base.save_json(
        {
            "method_label": args.method_label,
            "candidate_root": str(args.candidate_root),
            "budget_rows": len(rows),
            "completed_jobs": len({(row["graph_name"], row["diffusion"]) for row in rows}),
            "failures": failures,
            "elapsed_seconds": time.perf_counter() - started,
            "summary_csv": str(summary_path),
        },
        args.output_dir / f"index_{args.method_label}.json",
    )
    LOGGER.info("Finished: %d rows, %d failure(s).", len(rows), len(failures))
    LOGGER.info("Summary: %s", summary_path)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
