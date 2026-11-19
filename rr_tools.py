#!/usr/bin/env python3
"""Reverse-reachable-set utilities for verified influence-maximization search.

The processed graph format is the JSON format produced by preprocess_graphs.py.
This module uses Python integer bitsets for compact and fast RR-set coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import gzip
import json
import math
import pickle
import random


@dataclass(frozen=True)
class SpreadEstimate:
    spread: float
    coverage_ratio: float
    covered_rr_sets: int
    num_rr_sets: int
    standard_error: float
    ci95_low: float
    ci95_high: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "spread": self.spread,
            "coverage_ratio": self.coverage_ratio,
            "covered_rr_sets": self.covered_rr_sets,
            "num_rr_sets": self.num_rr_sets,
            "standard_error": self.standard_error,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
        }


@dataclass(frozen=True)
class GraphView:
    name: str
    num_nodes: int
    num_edges: int
    edges: tuple[tuple[int, int], ...]
    reverse_adjacency: tuple[tuple[int, ...], ...]
    in_degree: tuple[int, ...]
    out_degree: tuple[int, ...]

    @classmethod
    def from_processed_json(cls, graph: dict[str, Any]) -> "GraphView":
        n = int(graph["num_nodes"])
        edges = tuple((int(u), int(v)) for u, v in graph["edges"])

        reverse: list[list[int]] = [[] for _ in range(n + 1)]
        out_degree = [0] * (n + 1)
        in_degree = [0] * (n + 1)
        for u, v in edges:
            if not (1 <= u <= n and 1 <= v <= n):
                raise ValueError(f"Edge ({u}, {v}) is outside 1..{n}.")
            reverse[v].append(u)
            out_degree[u] += 1
            in_degree[v] += 1

        for node in range(1, n + 1):
            reverse[node].sort()

        return cls(
            name=str(graph.get("name", "graph")),
            num_nodes=n,
            num_edges=len(edges),
            edges=edges,
            reverse_adjacency=tuple(tuple(x) for x in reverse),
            in_degree=tuple(in_degree),
            out_degree=tuple(out_degree),
        )


def load_processed_graph(path: Path) -> tuple[dict[str, Any], GraphView]:
    with path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    return graph, GraphView.from_processed_json(graph)


def parse_diffusion_name(diffusion_name: str) -> tuple[str, float | None]:
    if diffusion_name == "weighted_cascade":
        return "weighted_cascade", None
    prefix = "uniform_ic_p"
    if diffusion_name.startswith(prefix):
        try:
            p = float(diffusion_name[len(prefix) :])
        except ValueError as exc:
            raise ValueError(f"Cannot parse diffusion name: {diffusion_name}") from exc
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Uniform IC probability must be in [0,1], got {p}.")
        return "uniform", p
    raise ValueError(
        f"Unsupported diffusion '{diffusion_name}'. Expected weighted_cascade "
        "or a name such as uniform_ic_p0.1."
    )


def edge_activation_probability(
    graph: GraphView,
    target: int,
    diffusion_kind: str,
    uniform_p: float | None,
) -> float:
    if diffusion_kind == "uniform":
        assert uniform_p is not None
        return uniform_p
    if diffusion_kind == "weighted_cascade":
        degree = graph.in_degree[target]
        return 0.0 if degree <= 0 else 1.0 / degree
    raise ValueError(f"Unknown diffusion kind: {diffusion_kind}")


def sample_one_rr_set(
    graph: GraphView,
    rng: random.Random,
    diffusion_kind: str,
    uniform_p: float | None,
) -> tuple[int, ...]:
    """Sample one RR set by traversing a sampled live-edge graph backwards."""
    root = rng.randint(1, graph.num_nodes)
    visited = {root}
    queue = [root]
    cursor = 0

    while cursor < len(queue):
        target = queue[cursor]
        cursor += 1
        p = edge_activation_probability(graph, target, diffusion_kind, uniform_p)
        if p <= 0.0:
            continue
        for predecessor in graph.reverse_adjacency[target]:
            if predecessor in visited:
                continue
            if p >= 1.0 or rng.random() < p:
                visited.add(predecessor)
                queue.append(predecessor)

    return tuple(sorted(visited))


def sample_rr_sets(
    graph: GraphView,
    num_rr_sets: int,
    diffusion_name: str,
    seed: int,
) -> list[tuple[int, ...]]:
    if num_rr_sets <= 0:
        raise ValueError("num_rr_sets must be positive.")
    diffusion_kind, uniform_p = parse_diffusion_name(diffusion_name)
    rng = random.Random(seed)
    return [
        sample_one_rr_set(graph, rng, diffusion_kind, uniform_p)
        for _ in range(num_rr_sets)
    ]


def save_rr_cache(
    path: Path,
    *,
    graph: GraphView,
    diffusion_name: str,
    seed: int,
    rr_sets: Sequence[Sequence[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "graph_name": graph.name,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "diffusion": diffusion_name,
        "seed": seed,
        "num_rr_sets": len(rr_sets),
        "rr_sets": [tuple(map(int, rr)) for rr in rr_sets],
    }
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_rr_cache(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "rr_sets" not in payload:
        raise ValueError(f"Invalid RR cache: {path}")
    return payload


def load_or_create_rr_sets(
    path: Path,
    *,
    graph: GraphView,
    diffusion_name: str,
    num_rr_sets: int,
    seed: int,
    overwrite: bool = False,
) -> list[tuple[int, ...]]:
    if path.exists() and not overwrite:
        payload = load_rr_cache(path)
        expected = {
            "graph_name": graph.name,
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "diffusion": diffusion_name,
            "seed": seed,
            "num_rr_sets": num_rr_sets,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"RR cache metadata mismatch for {path}: {mismatches}. "
                "Use --overwrite-rr-cache or choose a different cache directory."
            )
        return [tuple(map(int, rr)) for rr in payload["rr_sets"]]

    rr_sets = sample_rr_sets(
        graph=graph,
        num_rr_sets=num_rr_sets,
        diffusion_name=diffusion_name,
        seed=seed,
    )
    save_rr_cache(
        path,
        graph=graph,
        diffusion_name=diffusion_name,
        seed=seed,
        rr_sets=rr_sets,
    )
    return rr_sets


def build_node_rr_bitsets(
    rr_sets: Sequence[Sequence[int]],
    num_nodes: int,
) -> list[int]:
    """Return bitsets where bit i indicates that node occurs in RR set i."""
    bitsets = [0] * (num_nodes + 1)
    for rr_index, rr_set in enumerate(rr_sets):
        bit = 1 << rr_index
        for node in rr_set:
            node = int(node)
            if not (1 <= node <= num_nodes):
                raise ValueError(f"RR node {node} is outside 1..{num_nodes}.")
            bitsets[node] |= bit
    return bitsets


def covered_bitset(seed_set: Iterable[int], node_rr_bitsets: Sequence[int]) -> int:
    covered = 0
    for node in seed_set:
        covered |= node_rr_bitsets[int(node)]
    return covered


def estimate_spread_from_covered(
    covered: int,
    *,
    num_nodes: int,
    num_rr_sets: int,
) -> SpreadEstimate:
    if num_rr_sets <= 0:
        raise ValueError("num_rr_sets must be positive.")
    count = covered.bit_count()
    ratio = count / num_rr_sets
    spread = num_nodes * ratio
    se = num_nodes * math.sqrt(max(ratio * (1.0 - ratio), 0.0) / num_rr_sets)
    delta = 1.96 * se
    return SpreadEstimate(
        spread=spread,
        coverage_ratio=ratio,
        covered_rr_sets=count,
        num_rr_sets=num_rr_sets,
        standard_error=se,
        ci95_low=max(0.0, spread - delta),
        ci95_high=min(float(num_nodes), spread + delta),
    )


def estimate_spread(
    seed_set: Iterable[int],
    *,
    node_rr_bitsets: Sequence[int],
    num_nodes: int,
    num_rr_sets: int,
) -> SpreadEstimate:
    return estimate_spread_from_covered(
        covered_bitset(seed_set, node_rr_bitsets),
        num_nodes=num_nodes,
        num_rr_sets=num_rr_sets,
    )


def coverage_jaccard(bitset_a: int, bitset_b: int) -> float:
    union = (bitset_a | bitset_b).bit_count()
    if union == 0:
        return 0.0
    return (bitset_a & bitset_b).bit_count() / union


def marginal_rr_gain(
    candidate: int,
    current_covered: int,
    node_rr_bitsets: Sequence[int],
) -> int:
    return (node_rr_bitsets[int(candidate)] & ~current_covered).bit_count()


def union_without_node(
    seed_set: Sequence[int],
    removed_node: int,
    node_rr_bitsets: Sequence[int],
) -> int:
    covered = 0
    removed_once = False
    for node in seed_set:
        if int(node) == int(removed_node) and not removed_once:
            removed_once = True
            continue
        covered |= node_rr_bitsets[int(node)]
    return covered


def leave_one_out_rr_loss(
    seed_set: Sequence[int],
    removed_node: int,
    node_rr_bitsets: Sequence[int],
) -> int:
    full = covered_bitset(seed_set, node_rr_bitsets)
    without = union_without_node(seed_set, removed_node, node_rr_bitsets)
    return full.bit_count() - without.bit_count()


def swap_rr_delta(
    seed_set: Sequence[int],
    remove_node: int,
    add_node: int,
    node_rr_bitsets: Sequence[int],
) -> tuple[int, int]:
    """Return (delta RR coverage, new covered bitset)."""
    old_covered = covered_bitset(seed_set, node_rr_bitsets)
    without = union_without_node(seed_set, remove_node, node_rr_bitsets)
    new_covered = without | node_rr_bitsets[int(add_node)]
    return new_covered.bit_count() - old_covered.bit_count(), new_covered
