#!/usr/bin/env python3
"""Build equal-size candidate-pool variants for IM ablation experiments.

The script reads aggregate.json files produced by generate_candidates_qwen35.py
and writes compatible aggregate.json trees for alternative candidate sources:

* qwen_multi  : multi-permutation consensus ranking
* qwen_single : one fixed permutation ranking
* out_degree  : descending directed out-degree
* pagerank    : directed PageRank
* random      : deterministic random ranking

All sources are truncated to the same pool size so downstream verified-search
comparisons do not confound candidate quality with candidate-pool cardinality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from rr_tools import GraphView, load_processed_graph  # noqa: E402
from verified_search_fixed_budget import resolve_graph_path  # noqa: E402

LOGGER = logging.getLogger("candidate_ablation_builder")
SUPPORTED_SOURCES = ("qwen_multi", "qwen_single", "out_degree", "pagerank", "random")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *map(str, parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def reciprocal_log_scores(ranking: list[int]) -> dict[int, float]:
    return {
        int(node): 1.0 / math.log2(rank + 1.0)
        for rank, node in enumerate(ranking, start=1)
    }


def deduplicate_valid(nodes: Iterable[Any], num_nodes: int) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for raw in nodes:
        try:
            node = int(raw)
        except (TypeError, ValueError):
            continue
        if not 1 <= node <= num_nodes or node in seen:
            continue
        seen.add(node)
        result.append(node)
    return result


def directed_pagerank(
    graph: GraphView,
    *,
    damping: float = 0.85,
    tolerance: float = 1e-12,
    max_iterations: int = 300,
) -> dict[int, float]:
    """Compute deterministic PageRank for a directed GraphView."""
    n = graph.num_nodes
    if n <= 0:
        return {}

    outgoing: list[list[int]] = [[] for _ in range(n + 1)]
    for source, target in graph.edges:
        outgoing[source].append(target)

    rank = [0.0] + [1.0 / n] * n
    teleport = (1.0 - damping) / n

    for _ in range(max_iterations):
        next_rank = [0.0] + [teleport] * n
        sink_mass = sum(rank[node] for node in range(1, n + 1) if not outgoing[node])
        sink_share = damping * sink_mass / n
        if sink_share:
            for node in range(1, n + 1):
                next_rank[node] += sink_share

        for source in range(1, n + 1):
            neighbors = outgoing[source]
            if not neighbors:
                continue
            contribution = damping * rank[source] / len(neighbors)
            for target in neighbors:
                next_rank[target] += contribution

        error = sum(abs(next_rank[node] - rank[node]) for node in range(1, n + 1))
        rank = next_rank
        if error <= tolerance:
            break

    total = sum(rank[1:])
    if total > 0:
        rank = [0.0] + [value / total for value in rank[1:]]
    return {node: rank[node] for node in range(1, n + 1)}


def first_or_fixed_permutation_ranking(
    aggregate_path: Path,
    *,
    permutation_index: int,
    fallback: str,
    num_nodes: int,
) -> tuple[list[int], str]:
    runs_dir = aggregate_path.parent / "runs"
    requested = runs_dir / f"perm_{permutation_index:03d}.json"
    candidates = [requested]
    if fallback == "first_valid":
        candidates.extend(path for path in sorted(runs_dir.glob("perm_*.json")) if path != requested)

    diagnostics: list[str] = []
    for path in candidates:
        if not path.exists():
            diagnostics.append(f"missing:{path.name}")
            continue
        run = load_json(path)
        ranking = deduplicate_valid(run.get("internal_candidates", []), num_nodes)
        if ranking:
            return ranking, path.name
        diagnostics.append(f"empty:{path.name}:{run.get('status')}")

    raise RuntimeError(
        "No usable single-permutation ranking. " + ", ".join(diagnostics[:10])
    )


def source_ranking(
    source: str,
    *,
    aggregate: dict[str, Any],
    aggregate_path: Path,
    graph: GraphView,
    base_seed: int,
    single_permutation_index: int,
    single_fallback: str,
) -> tuple[list[int], dict[int, float], dict[str, Any]]:
    n = graph.num_nodes
    metadata: dict[str, Any] = {}

    if source == "qwen_multi":
        ranking = deduplicate_valid(aggregate.get("candidate_pool_ids", []), n)
        native_scores: dict[int, float] = {}
        for item in aggregate.get("candidate_pool", []):
            try:
                native_scores[int(item["internal_node"])] = float(item["consensus_score"])
            except (KeyError, TypeError, ValueError):
                continue
        metadata["original_pool_size"] = len(ranking)
        return ranking, native_scores, metadata

    if source == "qwen_single":
        ranking, selected_run = first_or_fixed_permutation_ranking(
            aggregate_path,
            permutation_index=single_permutation_index,
            fallback=single_fallback,
            num_nodes=n,
        )
        metadata["selected_run"] = selected_run
        return ranking, reciprocal_log_scores(ranking), metadata

    if source == "out_degree":
        ranking = sorted(
            range(1, n + 1),
            key=lambda node: (
                -graph.out_degree[node],
                -graph.in_degree[node],
                node,
            ),
        )
        native = {node: float(graph.out_degree[node]) for node in ranking}
        return ranking, native, metadata

    if source == "pagerank":
        native = directed_pagerank(graph)
        ranking = sorted(
            range(1, n + 1),
            key=lambda node: (-native[node], -graph.out_degree[node], node),
        )
        return ranking, native, metadata

    if source == "random":
        ranking = list(range(1, n + 1))
        seed = stable_seed(
            base_seed,
            aggregate.get("graph_name"),
            aggregate.get("diffusion"),
            "random_candidate_source",
        )
        random.Random(seed).shuffle(ranking)
        metadata["random_seed"] = seed
        return ranking, reciprocal_log_scores(ranking), metadata

    raise ValueError(f"Unsupported source: {source}")


def build_variant(
    source: str,
    *,
    aggregate: dict[str, Any],
    aggregate_path: Path,
    input_root: Path,
    output_root: Path,
    pool_size: int,
    score_mode: str,
    base_seed: int,
    single_permutation_index: int,
    single_fallback: str,
    overwrite: bool,
) -> Path:
    graph_path = resolve_graph_path(str(aggregate["graph_file"]), input_root)
    _, graph = load_processed_graph(graph_path)

    ranking, native_scores, source_metadata = source_ranking(
        source,
        aggregate=aggregate,
        aggregate_path=aggregate_path,
        graph=graph,
        base_seed=base_seed,
        single_permutation_index=single_permutation_index,
        single_fallback=single_fallback,
    )

    ranking = deduplicate_valid(ranking, graph.num_nodes)
    if pool_size > 0:
        ranking = ranking[: min(pool_size, len(ranking))]
    budgets = [int(value) for value in aggregate.get("budgets", [10, 20, 30])]
    if len(ranking) < max(budgets):
        raise RuntimeError(
            f"Source {source} produced only {len(ranking)} nodes, fewer than "
            f"maximum budget {max(budgets)} for {aggregate_path}."
        )

    rank_scores = reciprocal_log_scores(ranking)
    if score_mode == "native" and source != "random":
        scores = {
            node: float(native_scores.get(node, rank_scores[node]))
            for node in ranking
        }
        if max(scores.values(), default=0.0) <= 0.0:
            scores = rank_scores
    else:
        scores = rank_scores

    relative = aggregate_path.relative_to(input_root)
    destination = output_root / source / relative
    if destination.exists() and not overwrite:
        return destination

    pool_rows = []
    for rank, node in enumerate(ranking, start=1):
        pool_rows.append(
            {
                "internal_node": node,
                "original_node": node,
                "consensus_score": scores[node],
                "source_rank": rank,
                "candidate_source": source,
            }
        )

    variant = dict(aggregate)
    variant.update(
        {
            "candidate_pool_size": len(ranking),
            "candidate_pool_ids": ranking,
            "candidate_pool": pool_rows,
            "consensus_seed_baselines": {
                str(budget): ranking[:budget] for budget in budgets
            },
            "backend": "candidate_source_ablation",
            "model_name": source,
            "ablation": {
                "candidate_source": source,
                "pool_size": len(ranking),
                "requested_pool_size": pool_size,
                "score_mode": score_mode,
                "base_seed": base_seed,
                "source_metadata": source_metadata,
                "original_aggregate_file": str(aggregate_path),
            },
        }
    )
    save_json(variant, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create equal-size candidate-source ablation aggregate trees."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=SCRIPT_DIR / "candidate_outputs_qwen35_9b",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "candidate_ablation_inputs",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SUPPORTED_SOURCES,
        default=list(SUPPORTED_SOURCES),
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=60,
        help="Equal candidate-pool size for all sources; 0 keeps all available nodes.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("rank", "native"),
        default="rank",
        help="Use reciprocal-log rank scores for fairness, or source-native scores.",
    )
    parser.add_argument("--single-permutation-index", type=int, default=0)
    parser.add_argument(
        "--single-fallback",
        choices=("none", "first_valid"),
        default="first_valid",
    )
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    if not args.input_root.exists():
        raise FileNotFoundError(args.input_root)
    if args.pool_size < 0:
        raise ValueError("--pool-size cannot be negative.")
    if args.single_permutation_index < 0:
        raise ValueError("--single-permutation-index cannot be negative.")
    if args.clean_output and args.output_root.exists():
        shutil.rmtree(args.output_root)

    aggregate_paths = sorted(args.input_root.rglob("aggregate.json"))
    if not aggregate_paths:
        raise RuntimeError(f"No aggregate.json files under {args.input_root}")

    completed: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for source in args.sources:
        for aggregate_path in aggregate_paths:
            try:
                destination = build_variant(
                    source,
                    aggregate=load_json(aggregate_path),
                    aggregate_path=aggregate_path,
                    input_root=args.input_root,
                    output_root=args.output_root,
                    pool_size=args.pool_size,
                    score_mode=args.score_mode,
                    base_seed=args.base_seed,
                    single_permutation_index=args.single_permutation_index,
                    single_fallback=args.single_fallback,
                    overwrite=args.overwrite,
                )
                completed.append(
                    {
                        "source": source,
                        "input": str(aggregate_path),
                        "output": str(destination),
                    }
                )
            except Exception as exc:
                LOGGER.exception("Failed source=%s aggregate=%s", source, aggregate_path)
                failures.append(
                    {
                        "source": source,
                        "aggregate": str(aggregate_path),
                        "error": str(exc),
                    }
                )

    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "sources": args.sources,
        "pool_size": args.pool_size,
        "score_mode": args.score_mode,
        "completed": len(completed),
        "failures": failures,
        "items": completed,
    }
    save_json(manifest, args.output_root / "ablation_manifest.json")

    LOGGER.info("Created %d aggregate variants; %d failure(s).", len(completed), len(failures))
    LOGGER.info("Manifest: %s", args.output_root / "ablation_manifest.json")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
