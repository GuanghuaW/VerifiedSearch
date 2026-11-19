#!/usr/bin/env python3
"""Re-run the paper's direct-LLM majority-vote baseline on the current protocol.

The original manuscript used LLaMA 3.1 8B, few-shot/step-by-step prompting,
10 stochastic runs, majority voting, and a best-run result. This updated script
preserves that high-level design while making the experiment auditable and
comparable to VerifiedSearch:

* current 27 processed graphs;
* Uniform IC p=0.1 and Weighted Cascade;
* budgets 10/20/30;
* identical independent 10,000-RR-set evaluation caches;
* best-of-10 selected on separate tool RR sets, never on evaluation RR sets;
* raw model outputs, parser diagnostics, and seeds saved for every run.

The original paper did not report every decoding/parser hyperparameter, so this
is an updated reproducible rerun rather than a byte-identical reconstruction.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from baseline_common import (  # noqa: E402
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
from rr_tools import build_node_rr_bitsets, estimate_spread, load_or_create_rr_sets  # noqa: E402

LOGGER = logging.getLogger("direct_llm_mv")


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "model"


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    if re.search(r"<think>", cleaned, flags=re.I):
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.I | re.S)
    return cleaned.strip()


def parse_nodes(text: str, *, num_nodes: int, desired_length: int) -> tuple[list[int], dict[str, Any]]:
    cleaned = strip_thinking(text)
    fragments = [cleaned]
    left, right = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= left < right:
        fragments.append(cleaned[left : right + 1])
    left, right = cleaned.find("["), cleaned.rfind("]")
    if 0 <= left < right:
        fragments.append(cleaned[left : right + 1])

    raw_items: list[Any] = []
    parser = "none"
    keys = ("seed_nodes", "seeds", "ranked_candidates", "nodes", "answer")
    for fragment in fragments:
        for loader_name, loader in (("json", json.loads), ("literal", ast.literal_eval)):
            try:
                payload = loader(fragment)
            except Exception:
                continue
            value: Any = payload
            if isinstance(payload, dict):
                value = None
                for key in keys:
                    if isinstance(payload.get(key), list):
                        value = payload[key]
                        break
            if isinstance(value, (list, tuple)):
                raw_items = list(value)
                parser = loader_name
                break
        if raw_items:
            break

    if not raw_items:
        key_pattern = r"(?:seed_nodes|seeds|ranked_candidates|nodes|answer)\s*[:=]\s*\[([^\]]*)\]"
        match = re.search(key_pattern, cleaned, flags=re.I | re.S)
        if match:
            raw_items = re.findall(r"-?\d+", match.group(1))
            parser = "key_regex"

    if not raw_items:
        hash_nodes = re.findall(r"#\s*(\d+)", cleaned)
        if hash_nodes:
            raw_items = hash_nodes
            parser = "hash_regex"

    result: list[int] = []
    invalid: list[Any] = []
    duplicates: list[int] = []
    seen: set[int] = set()
    for value in raw_items:
        match = re.search(r"-?\d+", str(value))
        if not match:
            invalid.append(value)
            continue
        node = int(match.group())
        if not 1 <= node <= num_nodes:
            invalid.append(value)
            continue
        if node in seen:
            duplicates.append(node)
            continue
        seen.add(node)
        result.append(node)
        if len(result) == desired_length:
            break

    return result, {
        "parser": parser,
        "parsed_count": len(result),
        "desired_length": desired_length,
        "invalid_items": invalid,
        "duplicate_items": duplicates,
    }


def diffusion_description(diffusion: str) -> str:
    if diffusion == "weighted_cascade":
        return (
            "Weighted Cascade Independent Cascade: for edge u->v, activation "
            "probability is 1 / in_degree(v)."
        )
    if diffusion.startswith("uniform_ic_p"):
        p = diffusion.removeprefix("uniform_ic_p")
        return f"Uniform Independent Cascade: every directed edge activates with probability {p}."
    raise ValueError(f"Unsupported diffusion: {diffusion}")


def make_prompt(
    *,
    graph_name: str,
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    diffusion: str,
    requested_length: int,
    prompt_style: str,
) -> str:
    edge_text = ", ".join(f"({u},{v})" for u, v in edges)
    demo = (
        "Example: for edges [(1,2),(2,6),(3,4),(4,6),(5,6),(6,7),"
        "(7,8),(5,9),(9,10)] and budget 3, a coverage-oriented answer is "
        '{"seed_nodes":[1,3,5]}. '
    )
    rationale = ""
    if prompt_style == "paper_updated":
        rationale = (
            "Use the example as a few-shot demonstration. Analyze reachability, "
            "outgoing influence, and overlap among selected seeds. Think through "
            "the graph internally before answering. "
        )
    elif prompt_style == "json_only":
        demo = ""
        rationale = "Optimize joint influence and avoid redundant seeds. "
    else:
        raise ValueError(prompt_style)

    return (
        f"{demo}{rationale}\n"
        "Task: choose an ordered list of distinct seed nodes for influence maximization.\n"
        f"Graph name: {graph_name}\n"
        f"Number of nodes: {num_nodes}; valid IDs are 1 through {num_nodes}.\n"
        f"Diffusion model: {diffusion_description(diffusion)}\n"
        f"Return exactly {requested_length} distinct node IDs, ranked best first.\n"
        f"Directed edges: [{edge_text}]\n\n"
        "Return ONLY valid JSON in this exact form:\n"
        f'{{"seed_nodes":[n1,n2,...,n{requested_length}]}}'
    )


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
        raise RuntimeError(f"Majority vote has only {len(ordered)} unique nodes for budget {budget}")
    return ordered[:budget]


def fill_by_out_degree(
    parsed: Sequence[int],
    *,
    graph: Any,
    desired_length: int,
) -> list[int]:
    result = list(dict.fromkeys(map(int, parsed)))
    used = set(result)
    ranking = sorted(
        range(1, graph.num_nodes + 1),
        key=lambda node: (-graph.out_degree[node], -graph.in_degree[node], node),
    )
    for node in ranking:
        if node not in used:
            result.append(node)
            used.add(node)
        if len(result) == desired_length:
            break
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("VerifiedSearch/candidate_outputs_qwen35_9b"),
        help="Used to enumerate the same 54 graph-diffusion instances.",
    )
    parser.add_argument(
        "--rr-cache-dir",
        type=Path,
        default=Path("VerifiedSearch/rr_cache_complete"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("VerifiedSearch/updated_baseline_outputs/direct_llm"),
    )
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument(
        "--generation-mode",
        choices=("per_budget", "ranked_prefix"),
        default="per_budget",
        help=(
            "per_budget most closely follows the manuscript; ranked_prefix makes one "
            "top-30 call per run and derives smaller budgets by prefix."
        ),
    )
    parser.add_argument("--prompt-style", choices=("paper_updated", "json_only"), default="paper_updated")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--invalid-policy", choices=("skip", "out_degree_fill"), default="skip")
    parser.add_argument("--min-valid-runs", type=int, default=5)
    parser.add_argument("--tool-rr-sets", type=int, default=500)
    parser.add_argument("--eval-rr-sets", type=int, default=10000)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--graphs", nargs="*", default=None)
    parser.add_argument("--dataset-types", nargs="+", choices=("synthetic", "real"), default=None)
    parser.add_argument(
        "--diffusions",
        nargs="+",
        choices=("uniform_ic_p0.1", "weighted_cascade"),
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-rr-cache", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.runs <= 0 or args.max_retries < 0:
        raise SystemExit("--runs must be positive and --max-retries nonnegative")
    budgets = sorted(set(map(int, args.budgets)))
    if not budgets or budgets[0] <= 0:
        raise SystemExit("Budgets must be positive")

    try:
        from generate_candidates_qwen35 import HuggingFaceBackend
    except ImportError as exc:
        raise SystemExit(
            "Place this script beside generate_candidates_qwen35.py and rr_tools.py in VerifiedSearch."
        ) from exc

    backend = HuggingFaceBackend(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        enable_thinking=args.enable_thinking,
    )

    candidate_root = args.candidate_root.resolve()
    rr_cache_dir = args.rr_cache_dir.resolve()
    model_key = safe_slug(args.model_name)
    output_dir = args.output_dir.resolve() / model_key / args.generation_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_paths = list(
        iter_aggregate_paths(
            candidate_root,
            graph_names=set(args.graphs) if args.graphs else None,
            dataset_types=set(args.dataset_types) if args.dataset_types else None,
            diffusions=set(args.diffusions) if args.diffusions else None,
        )
    )
    if not aggregate_paths:
        raise SystemExit("No aggregate files matched filters")

    mv_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for item_index, aggregate_path in enumerate(aggregate_paths, start=1):
        aggregate, graph_path, graph, eval_bitsets, _ = load_evaluation_context(
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
        LOGGER.info("[%d/%d] %s | %s", item_index, len(aggregate_paths), graph_name, diffusion)

        tool_seed = stable_seed(args.base_seed, graph_name, diffusion, "tool_rr")
        cache_dir = rr_cache_dir / dataset_type / graph_name / diffusion
        tool_cache = cache_dir / f"tool_rr_{args.tool_rr_sets}_seed{tool_seed}.pkl.gz"
        tool_rr = load_or_create_rr_sets(
            tool_cache,
            graph=graph,
            diffusion_name=diffusion,
            num_rr_sets=args.tool_rr_sets,
            seed=tool_seed,
            overwrite=args.overwrite_rr_cache,
        )
        tool_bitsets = build_node_rr_bitsets(tool_rr, graph.num_nodes)

        instance_dir = output_dir / dataset_type / graph_name / diffusion
        instance_file = instance_dir / "direct_llm_runs.json"
        if instance_file.exists() and not args.overwrite:
            payload = load_json(instance_file)
            stored = payload.get("configuration", {})
            requested = {
                "runs": args.runs,
                "budgets": budgets,
                "generation_mode": args.generation_mode,
                "prompt_style": args.prompt_style,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "max_retries": args.max_retries,
                "invalid_policy": args.invalid_policy,
                "base_seed": args.base_seed,
            }
            mismatches = {
                key: (stored.get(key), value)
                for key, value in requested.items()
                if stored.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    f"Cached direct-LLM configuration mismatch for {instance_file}: "
                    f"{mismatches}. Use --overwrite or a different --output-dir."
                )
            LOGGER.info("Reusing %s", instance_file)
        else:
            payload = {
                "version": 1,
                "model_name": args.model_name,
                "graph_name": graph_name,
                "dataset_type": dataset_type,
                "diffusion": diffusion,
                "graph_file": str(graph_path),
                "configuration": {
                    "runs": args.runs,
                    "budgets": budgets,
                    "generation_mode": args.generation_mode,
                    "prompt_style": args.prompt_style,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "max_retries": args.max_retries,
                    "invalid_policy": args.invalid_policy,
                    "base_seed": args.base_seed,
                },
                "runs_by_budget": {str(b): [] for b in budgets},
            }

            call_lengths = budgets if args.generation_mode == "per_budget" else [max(budgets)]
            generated_by_length: dict[int, list[dict[str, Any]]] = {}
            for requested_length in call_lengths:
                calls: list[dict[str, Any]] = []
                prompt = make_prompt(
                    graph_name=graph_name,
                    num_nodes=graph.num_nodes,
                    edges=graph.edges,
                    diffusion=diffusion,
                    requested_length=requested_length,
                    prompt_style=args.prompt_style,
                )
                for run_index in range(args.runs):
                    attempts: list[dict[str, Any]] = []
                    parsed: list[int] = []
                    used_fallback = False
                    total_wall = 0.0
                    for retry in range(args.max_retries + 1):
                        seed = stable_seed(
                            args.base_seed,
                            graph_name,
                            diffusion,
                            requested_length,
                            run_index,
                            retry,
                            "direct_llm",
                        )
                        retry_prompt = prompt
                        if retry:
                            retry_prompt += (
                                "\nCORRECTION: The previous output was invalid or incomplete. "
                                "Return only the required JSON with distinct valid IDs."
                            )
                        started = time.perf_counter()
                        generation = backend.generate(
                            retry_prompt,
                            seed=seed,
                            permutation={"num_nodes": graph.num_nodes, "edges": list(graph.edges)},
                            list_length=requested_length,
                        )
                        wall = time.perf_counter() - started
                        total_wall += wall
                        parsed, diagnostics = parse_nodes(
                            generation.text,
                            num_nodes=graph.num_nodes,
                            desired_length=requested_length,
                        )
                        attempts.append(
                            {
                                "retry": retry,
                                "seed": seed,
                                "raw_output": generation.text,
                                "generation_metadata": generation.metadata,
                                "wall_seconds": wall,
                                "parse_diagnostics": diagnostics,
                                "parsed_nodes": parsed,
                            }
                        )
                        if len(parsed) == requested_length:
                            break
                    if len(parsed) < requested_length and args.invalid_policy == "out_degree_fill":
                        parsed = fill_by_out_degree(
                            parsed,
                            graph=graph,
                            desired_length=requested_length,
                        )
                        used_fallback = True
                    valid = len(parsed) == requested_length
                    calls.append(
                        {
                            "run_index": run_index,
                            "requested_length": requested_length,
                            "valid": valid,
                            "used_fallback": used_fallback,
                            "seed_nodes": parsed,
                            "total_wall_seconds": total_wall,
                            "attempts": attempts,
                        }
                    )
                generated_by_length[requested_length] = calls

            for budget in budgets:
                source_length = budget if args.generation_mode == "per_budget" else max(budgets)
                for call in generated_by_length[source_length]:
                    copied = dict(call)
                    copied["seed_nodes"] = list(map(int, call["seed_nodes"][:budget]))
                    copied["requested_budget"] = budget
                    copied["valid"] = len(copied["seed_nodes"]) == budget
                    payload["runs_by_budget"][str(budget)].append(copied)
            save_json(payload, instance_file)

        for budget in budgets:
            calls = list(payload.get("runs_by_budget", {}).get(str(budget), []))
            valid_calls = [
                call for call in calls
                if call.get("valid") and len(call.get("seed_nodes", [])) >= budget
            ]
            if len(valid_calls) < args.min_valid_runs:
                failure = {
                    "graph_name": graph_name,
                    "diffusion": diffusion,
                    "budget": budget,
                    "valid_runs": len(valid_calls),
                    "required": args.min_valid_runs,
                    "instance_file": str(instance_file),
                }
                failures.append(failure)
                LOGGER.error("Insufficient valid runs: %s", failure)
                continue

            rankings = [
                validate_seed_set(call["seed_nodes"], budget=budget, num_nodes=graph.num_nodes)
                for call in valid_calls
            ]
            mv_seeds = majority_vote(rankings, budget)
            mv_metrics = evaluate_seeds(
                mv_seeds,
                graph=graph,
                eval_bitsets=eval_bitsets,
                eval_rr_sets=args.eval_rr_sets,
            )
            total_generation = sum(float(call.get("total_wall_seconds", 0.0)) for call in valid_calls)
            fallback_runs = sum(bool(call.get("used_fallback")) for call in valid_calls)
            mv_method = f"{model_key}_mv{args.runs}"
            mv_rows.append(
                make_summary_row(
                    method=mv_method,
                    aggregate=aggregate,
                    graph=graph,
                    budget=budget,
                    seeds=mv_seeds,
                    metrics=mv_metrics,
                    eval_rr_sets=args.eval_rr_sets,
                    runtime_seconds=total_generation,
                    source_file=str(instance_file),
                    notes=(
                        f"Majority vote over {len(valid_calls)}/{args.runs} valid runs; "
                        f"fallback runs={fallback_runs}; mode={args.generation_mode}."
                    ),
                )
            )

            # Best-of-10 is chosen only with tool RR sets, then evaluated independently.
            tool_scores: list[float] = []
            eval_scores: list[float] = []
            for call, seeds in zip(valid_calls, rankings):
                tool_estimate = estimate_spread(
                    seeds,
                    node_rr_bitsets=tool_bitsets,
                    num_nodes=graph.num_nodes,
                    num_rr_sets=args.tool_rr_sets,
                )
                eval_estimate = evaluate_seeds(
                    seeds,
                    graph=graph,
                    eval_bitsets=eval_bitsets,
                    eval_rr_sets=args.eval_rr_sets,
                )
                tool_scores.append(tool_estimate.spread)
                eval_scores.append(float(eval_estimate["spread"]))
                run_rows.append(
                    {
                        "method": f"{model_key}_run",
                        "graph_name": graph_name,
                        "dataset_type": dataset_type,
                        "diffusion": diffusion,
                        "budget": budget,
                        "run_index": call.get("run_index"),
                        "valid_runs": len(valid_calls),
                        "tool_spread": tool_estimate.spread,
                        "final_eval_spread": eval_estimate["spread"],
                        "normalized_final_spread": eval_estimate["normalized_spread"],
                        "runtime_seconds": call.get("total_wall_seconds", 0.0),
                        "used_fallback": call.get("used_fallback", False),
                        "seed_set": json.dumps(seeds),
                        "source_file": str(instance_file),
                    }
                )
            best_index = max(range(len(valid_calls)), key=lambda i: (tool_scores[i], -i))
            best_seeds = rankings[best_index]
            best_metrics = evaluate_seeds(
                best_seeds,
                graph=graph,
                eval_bitsets=eval_bitsets,
                eval_rr_sets=args.eval_rr_sets,
            )
            best_method = f"{model_key}_best{args.runs}_tool"
            best_rows.append(
                make_summary_row(
                    method=best_method,
                    aggregate=aggregate,
                    graph=graph,
                    budget=budget,
                    seeds=best_seeds,
                    metrics=best_metrics,
                    eval_rr_sets=args.eval_rr_sets,
                    runtime_seconds=total_generation,
                    source_file=str(instance_file),
                    notes=(
                        f"Best of {len(valid_calls)} valid runs selected using {args.tool_rr_sets} "
                        "tool RR sets; evaluation RR sets were not used for selection."
                    ),
                )
            )

    write_summary(mv_rows, output_dir / f"summary_{model_key}_mv{args.runs}.csv")
    write_summary(best_rows, output_dir / f"summary_{model_key}_best{args.runs}_tool.csv")

    # Run-level file has additional fields, so write separately.
    import csv
    run_path = output_dir / f"runs_{model_key}.csv"
    if run_rows:
        with run_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
            writer.writeheader()
            writer.writerows(run_rows)

    save_json(
        {
            "version": 1,
            "model_name": args.model_name,
            "configuration": {
                **{key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
                "candidate_root": str(candidate_root),
                "rr_cache_dir": str(rr_cache_dir),
                "output_dir": str(output_dir),
            },
            "mv_rows": len(mv_rows),
            "best_rows": len(best_rows),
            "run_rows": len(run_rows),
            "failures": failures,
        },
        output_dir / "index_direct_llm.json",
    )
    LOGGER.info("MV rows=%d; best rows=%d; failures=%d", len(mv_rows), len(best_rows), len(failures))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
