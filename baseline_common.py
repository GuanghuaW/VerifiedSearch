#!/usr/bin/env python3
"""Shared utilities for updated influence-maximization baselines.

These utilities deliberately reuse the same processed graphs and independent
10,000-RR-set evaluation caches as VerifiedSearch so that all reported spreads
are directly comparable.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from rr_tools import (
    GraphView,
    build_node_rr_bitsets,
    estimate_spread,
    load_or_create_rr_sets,
    load_processed_graph,
)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def resolve_path(raw: str | Path, *, anchors: Sequence[Path]) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    candidates = [path]
    for anchor in anchors:
        candidates.append(anchor / path)
        candidates.append(anchor.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = "\n  ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Cannot resolve path {raw!r}. Tried:\n  {rendered}")


def iter_aggregate_paths(
    candidate_root: Path,
    *,
    graph_names: set[str] | None = None,
    dataset_types: set[str] | None = None,
    diffusions: set[str] | None = None,
) -> Iterator[Path]:
    for path in sorted(candidate_root.rglob("aggregate.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if graph_names is not None and str(payload.get("graph_name")) not in graph_names:
            continue
        if dataset_types is not None and str(payload.get("dataset_type")) not in dataset_types:
            continue
        if diffusions is not None and str(payload.get("diffusion")) not in diffusions:
            continue
        yield path


def validate_seed_set(seeds: Iterable[int], *, budget: int, num_nodes: int) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in seeds:
        node = int(value)
        if not 1 <= node <= num_nodes:
            continue
        if node in seen:
            continue
        seen.add(node)
        result.append(node)
        if len(result) == budget:
            break
    if len(result) != budget:
        raise ValueError(
            f"Need exactly {budget} valid unique nodes in 1..{num_nodes}; got {len(result)}."
        )
    return result


def load_evaluation_context(
    aggregate_path: Path,
    *,
    candidate_root: Path,
    rr_cache_dir: Path,
    eval_rr_sets: int,
    base_seed: int,
    overwrite_rr_cache: bool = False,
) -> tuple[dict[str, Any], Path, GraphView, list[int], int]:
    aggregate = load_json(aggregate_path)
    graph_path = resolve_path(
        aggregate["graph_file"],
        anchors=(candidate_root.resolve(), aggregate_path.parent.resolve()),
    )
    _, graph = load_processed_graph(graph_path)
    graph_name = str(aggregate["graph_name"])
    dataset_type = str(aggregate["dataset_type"])
    diffusion = str(aggregate["diffusion"])

    eval_seed = stable_seed(base_seed, graph_name, diffusion, "eval_rr")
    cache_dir = rr_cache_dir / dataset_type / graph_name / diffusion
    eval_cache = cache_dir / f"eval_rr_{eval_rr_sets}_seed{eval_seed}.pkl.gz"
    eval_rr = load_or_create_rr_sets(
        eval_cache,
        graph=graph,
        diffusion_name=diffusion,
        num_rr_sets=eval_rr_sets,
        seed=eval_seed,
        overwrite=overwrite_rr_cache,
    )
    bitsets = build_node_rr_bitsets(eval_rr, graph.num_nodes)
    return aggregate, graph_path, graph, bitsets, eval_seed


def evaluate_seeds(
    seeds: Sequence[int],
    *,
    graph: GraphView,
    eval_bitsets: Sequence[int],
    eval_rr_sets: int,
) -> dict[str, float | int]:
    estimate = estimate_spread(
        seeds,
        node_rr_bitsets=eval_bitsets,
        num_nodes=graph.num_nodes,
        num_rr_sets=eval_rr_sets,
    )
    result = estimate.to_dict()
    result["normalized_spread"] = estimate.spread / graph.num_nodes
    return result


SUMMARY_FIELDS = [
    "method",
    "graph_name",
    "dataset_type",
    "diffusion",
    "budget",
    "num_nodes",
    "num_edges",
    "final_eval_spread",
    "normalized_final_spread",
    "eval_covered_rr_sets",
    "eval_rr_sets",
    "eval_standard_error",
    "eval_ci95_low",
    "eval_ci95_high",
    "runtime_seconds",
    "seed_set",
    "source_file",
    "notes",
]


def make_summary_row(
    *,
    method: str,
    aggregate: dict[str, Any],
    graph: GraphView,
    budget: int,
    seeds: Sequence[int],
    metrics: dict[str, float | int],
    eval_rr_sets: int,
    runtime_seconds: float,
    source_file: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "method": method,
        "graph_name": aggregate["graph_name"],
        "dataset_type": aggregate["dataset_type"],
        "diffusion": aggregate["diffusion"],
        "budget": int(budget),
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "final_eval_spread": metrics["spread"],
        "normalized_final_spread": metrics["normalized_spread"],
        "eval_covered_rr_sets": metrics["covered_rr_sets"],
        "eval_rr_sets": eval_rr_sets,
        "eval_standard_error": metrics["standard_error"],
        "eval_ci95_low": metrics["ci95_low"],
        "eval_ci95_high": metrics["ci95_high"],
        "runtime_seconds": float(runtime_seconds),
        "seed_set": json.dumps(list(map(int, seeds))),
        "source_file": source_file,
        "notes": notes,
    }


def write_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
