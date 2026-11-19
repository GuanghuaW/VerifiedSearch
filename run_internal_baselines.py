#!/usr/bin/env python3
"""Run directly comparable internal influence-maximization baselines.

All methods use the current processed graphs and are evaluated with the same
independent RR-set caches used by VerifiedSearch.

Methods
-------
qwen_multi
    Top-k prefix of the five-permutation Qwen consensus ranking.
qwen_single
    Top-k prefix of one fixed Qwen permutation ranking.
random
    Deterministic instance-specific random seed set.
out_degree
    Directed out-degree ranking.
weighted_out_degree
    Ranking by sum of outgoing edge activation probabilities.
reverse_pagerank
    PageRank on the reversed graph, aligning incoming PageRank mass with
    outward influence in the original graph.
degree_discount_ic
    Directed adaptation of DegreeDiscountIC. It is reported only for uniform
    IC instances because the classical formula assumes one global p.
weighted_degree_discount
    Directed one-hop expected-marginal heuristic supporting both uniform IC
    and Weighted Cascade.
rr_greedy
    Greedy maximum RR coverage using independent selection RR sets. This is an
    RIS baseline, not the published IMM or OPIM-C implementation.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from baseline_common import (  # noqa: E402
    evaluate_seeds,
    iter_aggregate_paths,
    load_evaluation_context,
    load_json,
    make_summary_row,
    resolve_path,
    stable_seed,
    validate_seed_set,
    write_summary,
)
from rr_tools import (  # noqa: E402
    build_node_rr_bitsets,
    edge_activation_probability,
    load_or_create_rr_sets,
    parse_diffusion_name,
)

LOGGER = logging.getLogger("internal_baselines_v2")

METHOD_NAMES = {
    "qwen_multi": "qwen_multi_direct",
    "qwen_single": "qwen_single_direct",
    "random": "random_seed_selection",
    "out_degree": "out_degree",
    "weighted_out_degree": "weighted_out_degree",
    "reverse_pagerank": "reverse_pagerank",
    "degree_discount_ic": "degree_discount_ic_directed",
    "weighted_degree_discount": "weighted_degree_discount",
    "rr_greedy": "rr_greedy",
}


def build_out_neighbors(num_nodes: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    neighbors = [[] for _ in range(num_nodes + 1)]
    for u, v in edges:
        neighbors[int(u)].append(int(v))
    for node in range(1, num_nodes + 1):
        neighbors[node].sort()
    return neighbors


def load_single_ranking(
    aggregate: dict[str, Any],
    aggregate_path: Path,
    candidate_root: Path,
    preferred_index: int,
) -> tuple[list[int], str]:
    run_files = list(aggregate.get("run_files", []))
    candidates: list[tuple[int, Path]] = []
    for i, raw in enumerate(run_files):
        try:
            path = resolve_path(raw, anchors=(candidate_root, aggregate_path.parent))
        except FileNotFoundError:
            path = aggregate_path.parent / "runs" / f"perm_{i:03d}.json"
        candidates.append((i, path))

    if not candidates:
        candidates = [
            (i, path)
            for i, path in enumerate(sorted((aggregate_path.parent / "runs").glob("perm_*.json")))
        ]

    ordered = sorted(candidates, key=lambda pair: (pair[0] != preferred_index, pair[0]))
    for _, path in ordered:
        if not path.exists():
            continue
        run = load_json(path)
        ranking = [int(x) for x in run.get("internal_candidates", [])]
        if ranking:
            return ranking, str(path)
    raise RuntimeError(f"No usable permutation ranking for {aggregate_path}")


def pagerank_reversed(
    *,
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    damping: float = 0.85,
    tolerance: float = 1e-12,
    max_iterations: int = 500,
) -> list[float]:
    """Return PageRank scores on the reverse of the original directed graph."""
    reversed_out = [[] for _ in range(num_nodes + 1)]
    for u, v in edges:
        reversed_out[int(v)].append(int(u))

    rank = [0.0] + [1.0 / num_nodes] * num_nodes
    teleport = (1.0 - damping) / num_nodes
    for _ in range(max_iterations):
        dangling = sum(rank[v] for v in range(1, num_nodes + 1) if not reversed_out[v])
        new_rank = [0.0] + [teleport + damping * dangling / num_nodes] * num_nodes
        for source in range(1, num_nodes + 1):
            nbrs = reversed_out[source]
            if not nbrs:
                continue
            contribution = damping * rank[source] / len(nbrs)
            for target in nbrs:
                new_rank[target] += contribution
        delta = sum(abs(new_rank[i] - rank[i]) for i in range(1, num_nodes + 1))
        rank = new_rank
        if delta <= tolerance:
            break
    return rank


def weighted_out_degree_scores(graph: Any, diffusion: str) -> list[float]:
    kind, uniform_p = parse_diffusion_name(diffusion)
    scores = [0.0] * (graph.num_nodes + 1)
    for u, v in graph.edges:
        scores[u] += edge_activation_probability(graph, v, kind, uniform_p)
    return scores


def degree_discount_ic_select(graph: Any, budget: int, p: float) -> list[int]:
    """Directed adaptation of the classical DegreeDiscountIC heuristic.

    d[v] is directed out-degree. t[v] counts already selected in-neighbors of v.
    The classical score is d[v] - 2t[v] - (d[v]-t[v])t[v]p.
    """
    out_neighbors = build_out_neighbors(graph.num_nodes, graph.edges)
    d = list(map(float, graph.out_degree))
    t = [0] * (graph.num_nodes + 1)
    dd = d[:]
    selected: list[int] = []
    selected_set: set[int] = set()

    for _ in range(budget):
        node = max(
            (v for v in range(1, graph.num_nodes + 1) if v not in selected_set),
            key=lambda v: (dd[v], graph.out_degree[v], -v),
        )
        selected.append(node)
        selected_set.add(node)
        for child in out_neighbors[node]:
            if child in selected_set:
                continue
            t[child] += 1
            dd[child] = (
                d[child]
                - 2.0 * t[child]
                - (d[child] - t[child]) * t[child] * p
            )
    return selected


def weighted_degree_discount_select(graph: Any, budget: int, diffusion: str) -> list[int]:
    """Greedy one-hop expected-marginal heuristic for a directed IC graph.

    q[v] approximates the probability that v has not already been directly
    activated by selected seeds. Candidate score is
      q[v] + sum_(v,w) p(v,w) q[w].
    This discounts direct overlap while supporting heterogeneous WC weights.
    """
    kind, uniform_p = parse_diffusion_name(diffusion)
    outgoing: list[list[tuple[int, float]]] = [[] for _ in range(graph.num_nodes + 1)]
    for u, v in graph.edges:
        probability = edge_activation_probability(graph, v, kind, uniform_p)
        outgoing[u].append((v, probability))

    q_not_direct = [1.0] * (graph.num_nodes + 1)
    q_not_direct[0] = 0.0
    selected: list[int] = []
    selected_set: set[int] = set()

    for _ in range(budget):
        best_node: int | None = None
        best_score = float("-inf")
        for node in range(1, graph.num_nodes + 1):
            if node in selected_set:
                continue
            score = q_not_direct[node]
            for target, probability in outgoing[node]:
                if target in selected_set:
                    continue
                score += probability * q_not_direct[target]
            if (
                score > best_score
                or (
                    score == best_score
                    and (
                        best_node is None
                        or graph.out_degree[node] > graph.out_degree[best_node]
                        or (
                            graph.out_degree[node] == graph.out_degree[best_node]
                            and node < best_node
                        )
                    )
                )
            ):
                best_node = node
                best_score = score

        if best_node is None:
            raise RuntimeError("Weighted degree discount could not select a node")
        selected.append(best_node)
        selected_set.add(best_node)
        q_not_direct[best_node] = 0.0
        for target, probability in outgoing[best_node]:
            if target not in selected_set:
                q_not_direct[target] *= 1.0 - probability

    return selected


def rr_greedy_select(bitsets: Sequence[int], num_nodes: int, budget: int) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()
    covered = 0
    for _ in range(budget):
        best_node: int | None = None
        best_gain = -1
        for node in range(1, num_nodes + 1):
            if node in selected_set:
                continue
            gain = (bitsets[node] & ~covered).bit_count()
            if gain > best_gain or (gain == best_gain and (best_node is None or node < best_node)):
                best_gain = gain
                best_node = node
        if best_node is None:
            raise RuntimeError("RR greedy could not select another node")
        selected.append(best_node)
        selected_set.add(best_node)
        covered |= bitsets[best_node]
    return selected


def build_parser() -> argparse.ArgumentParser:
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
        "--output-dir",
        type=Path,
        default=Path("VerifiedSearch/baseline_outputs_v2/internal"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_NAMES),
        default=list(METHOD_NAMES),
    )
    parser.add_argument("--eval-rr-sets", type=int, default=10000)
    parser.add_argument("--selection-rr-sets", type=int, default=10000)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--single-run-index", type=int, default=0)
    parser.add_argument("--graphs", nargs="*", default=None)
    parser.add_argument("--dataset-types", nargs="+", choices=("synthetic", "real"), default=None)
    parser.add_argument(
        "--diffusions",
        nargs="+",
        choices=("uniform_ic_p0.1", "weighted_cascade"),
        default=None,
    )
    parser.add_argument("--overwrite-rr-cache", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    candidate_root = args.candidate_root.resolve()
    rr_cache_dir = args.rr_cache_dir.resolve()
    output_dir = args.output_dir.resolve()
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
        raise SystemExit("No aggregate.json files matched the requested filters.")

    rows_by_method: dict[str, list[dict[str, Any]]] = {
        METHOD_NAMES[name]: [] for name in args.methods
    }

    for index, aggregate_path in enumerate(aggregate_paths, start=1):
        aggregate, graph_path, graph, eval_bitsets, _ = load_evaluation_context(
            aggregate_path,
            candidate_root=candidate_root,
            rr_cache_dir=rr_cache_dir,
            eval_rr_sets=args.eval_rr_sets,
            base_seed=args.base_seed,
            overwrite_rr_cache=args.overwrite_rr_cache,
        )
        graph_name = str(aggregate["graph_name"])
        diffusion = str(aggregate["diffusion"])
        LOGGER.info("[%d/%d] %s | %s", index, len(aggregate_paths), graph_name, diffusion)

        single_ranking: list[int] | None = None
        single_source = ""
        if "qwen_single" in args.methods:
            single_ranking, single_source = load_single_ranking(
                aggregate, aggregate_path, candidate_root, args.single_run_index
            )

        reverse_pr: list[float] | None = None
        if "reverse_pagerank" in args.methods:
            reverse_pr = pagerank_reversed(num_nodes=graph.num_nodes, edges=graph.edges)

        weighted_out_scores: list[float] | None = None
        if "weighted_out_degree" in args.methods:
            weighted_out_scores = weighted_out_degree_scores(graph, diffusion)

        selection_bitsets: list[int] | None = None
        if "rr_greedy" in args.methods:
            selection_seed = stable_seed(args.base_seed, graph_name, diffusion, "baseline_select_rr")
            cache_dir = rr_cache_dir / str(aggregate["dataset_type"]) / graph_name / diffusion
            selection_cache = cache_dir / (
                f"baseline_select_rr_{args.selection_rr_sets}_seed{selection_seed}.pkl.gz"
            )
            selection_rr = load_or_create_rr_sets(
                selection_cache,
                graph=graph,
                diffusion_name=diffusion,
                num_rr_sets=args.selection_rr_sets,
                seed=selection_seed,
                overwrite=args.overwrite_rr_cache,
            )
            selection_bitsets = build_node_rr_bitsets(selection_rr, graph.num_nodes)

        budgets = [int(x) for x in aggregate.get("budgets", [10, 20, 30])]
        for budget in budgets:
            selections: dict[str, tuple[list[int], str, str, float]] = {}

            if "qwen_multi" in args.methods:
                start = time.perf_counter()
                raw = aggregate.get("consensus_seed_baselines", {}).get(str(budget))
                if not raw:
                    raw = list(aggregate.get("candidate_pool_ids", []))[:budget]
                seeds = validate_seed_set(raw, budget=budget, num_nodes=graph.num_nodes)
                selections[METHOD_NAMES["qwen_multi"]] = (
                    seeds,
                    str(aggregate_path),
                    "Direct top-k prefix of five-permutation Qwen consensus ranking.",
                    time.perf_counter() - start,
                )

            if "qwen_single" in args.methods:
                assert single_ranking is not None
                start = time.perf_counter()
                seeds = validate_seed_set(single_ranking, budget=budget, num_nodes=graph.num_nodes)
                selections[METHOD_NAMES["qwen_single"]] = (
                    seeds,
                    single_source,
                    f"Direct top-k prefix of one Qwen permutation (preferred index {args.single_run_index}).",
                    time.perf_counter() - start,
                )

            if "random" in args.methods:
                start = time.perf_counter()
                rng = random.Random(stable_seed(args.base_seed, graph_name, diffusion, budget, "random"))
                seeds = rng.sample(range(1, graph.num_nodes + 1), budget)
                selections[METHOD_NAMES["random"]] = (
                    seeds,
                    str(graph_path),
                    "Deterministic random seed set with an instance-specific stable seed.",
                    time.perf_counter() - start,
                )

            if "out_degree" in args.methods:
                start = time.perf_counter()
                ranking = sorted(
                    range(1, graph.num_nodes + 1),
                    key=lambda node: (-graph.out_degree[node], -graph.in_degree[node], node),
                )
                selections[METHOD_NAMES["out_degree"]] = (
                    ranking[:budget],
                    str(graph_path),
                    "Directed out-degree; in-degree and node ID break ties.",
                    time.perf_counter() - start,
                )

            if "weighted_out_degree" in args.methods:
                assert weighted_out_scores is not None
                start = time.perf_counter()
                ranking = sorted(
                    range(1, graph.num_nodes + 1),
                    key=lambda node: (-weighted_out_scores[node], -graph.out_degree[node], node),
                )
                selections[METHOD_NAMES["weighted_out_degree"]] = (
                    ranking[:budget],
                    str(graph_path),
                    "Sum of outgoing activation probabilities under the current diffusion model.",
                    time.perf_counter() - start,
                )

            if "reverse_pagerank" in args.methods:
                assert reverse_pr is not None
                start = time.perf_counter()
                ranking = sorted(
                    range(1, graph.num_nodes + 1),
                    key=lambda node: (-reverse_pr[node], -graph.out_degree[node], node),
                )
                selections[METHOD_NAMES["reverse_pagerank"]] = (
                    ranking[:budget],
                    str(graph_path),
                    "PageRank computed on the reversed directed graph.",
                    time.perf_counter() - start,
                )

            if "degree_discount_ic" in args.methods and diffusion.startswith("uniform_ic_p"):
                start = time.perf_counter()
                probability = float(diffusion.removeprefix("uniform_ic_p"))
                seeds = degree_discount_ic_select(graph, budget, probability)
                selections[METHOD_NAMES["degree_discount_ic"]] = (
                    seeds,
                    str(graph_path),
                    "Directed adaptation of classical DegreeDiscountIC; uniform IC only.",
                    time.perf_counter() - start,
                )

            if "weighted_degree_discount" in args.methods:
                start = time.perf_counter()
                seeds = weighted_degree_discount_select(graph, budget, diffusion)
                selections[METHOD_NAMES["weighted_degree_discount"]] = (
                    seeds,
                    str(graph_path),
                    "Directed one-hop expected-marginal degree-discount heuristic.",
                    time.perf_counter() - start,
                )

            if "rr_greedy" in args.methods:
                assert selection_bitsets is not None
                start = time.perf_counter()
                seeds = rr_greedy_select(selection_bitsets, graph.num_nodes, budget)
                selections[METHOD_NAMES["rr_greedy"]] = (
                    seeds,
                    str(graph_path),
                    f"Greedy coverage over {args.selection_rr_sets} independent selection RR sets; not IMM/OPIM-C.",
                    time.perf_counter() - start,
                )

            for method, (seeds, source_file, notes, runtime) in selections.items():
                metrics = evaluate_seeds(
                    seeds,
                    graph=graph,
                    eval_bitsets=eval_bitsets,
                    eval_rr_sets=args.eval_rr_sets,
                )
                rows_by_method[method].append(
                    make_summary_row(
                        method=method,
                        aggregate=aggregate,
                        graph=graph,
                        budget=budget,
                        seeds=seeds,
                        metrics=metrics,
                        eval_rr_sets=args.eval_rr_sets,
                        runtime_seconds=runtime,
                        source_file=source_file,
                        notes=notes,
                    )
                )

    for method, rows in rows_by_method.items():
        path = output_dir / f"summary_{method}.csv"
        write_summary(rows, path)
        LOGGER.info("Wrote %d rows: %s", len(rows), path)

    index_payload = {
        "version": 2,
        "candidate_root": str(candidate_root),
        "rr_cache_dir": str(rr_cache_dir),
        "eval_rr_sets": args.eval_rr_sets,
        "selection_rr_sets": args.selection_rr_sets,
        "base_seed": args.base_seed,
        "methods": {method: len(rows) for method, rows in rows_by_method.items()},
        "note": "degree_discount_ic_directed is uniform-IC only and therefore has 81 rows in the full protocol.",
    }
    with (output_dir / "index_internal_baselines.json").open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
