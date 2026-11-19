#!/usr/bin/env python3
"""Generate controlled sparse directed graphs for VerifiedSearch scalability tests.

Output is NetworkX node-link compatible JSON and can be consumed by the
existing VerifiedSearch/preprocess_graphs.py pipeline.

Each graph has exactly m = round(edge_factor * n) directed edges. A directed
ring backbone guarantees participation of every node, and remaining edges are
added with a lightweight preferential-endpoint process to create heterogeneous
degrees. These graphs are a controlled scalability stress test, not an
additional real-world benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def generate_graph(n: int, m: int, seed: int) -> list[tuple[int, int]]:
    if n < 2:
        raise ValueError("n must be at least 2")
    if m < n:
        raise ValueError("m must be >= n because a ring backbone is created")
    if m > n * (n - 1):
        raise ValueError("m exceeds the maximum for a simple directed graph")

    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()

    # Directed ring backbone.
    for u in range(1, n + 1):
        v = 1 if u == n else u + 1
        edges.add((u, v))

    # Preferential endpoint pool.
    endpoint_pool = list(range(1, n + 1)) * 2
    attempts = 0
    max_attempts = max(100_000, 100 * (m - len(edges) + 1))

    while len(edges) < m:
        attempts += 1
        if attempts <= max_attempts:
            u = rng.choice(endpoint_pool)
            v = rng.choice(endpoint_pool)
        else:
            # Deterministic uniform fallback.
            u = rng.randint(1, n)
            v = rng.randint(1, n)

        if u == v or (u, v) in edges:
            continue

        edges.add((u, v))
        endpoint_pool.append(u)
        endpoint_pool.append(v)

    return sorted(edges)


def write_node_link_json(
    output_path: Path,
    *,
    n: int,
    edges: Iterable[tuple[int, int]],
    generator_seed: int,
    edge_factor: float,
) -> None:
    edge_list = list(edges)
    payload = {
        "directed": True,
        "multigraph": False,
        "graph": {
            "generator": "ring_plus_preferential_sparse",
            "generator_seed": generator_seed,
            "edge_factor": edge_factor,
            "purpose": "VerifiedSearch scalability stress test",
        },
        "nodes": [{"id": node} for node in range(1, n + 1)],
        "links": [
            {"source": int(u), "target": int(v)}
            for u, v in edge_list
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sdata_scalability"),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2000, 5000, 10000, 20000],
    )
    parser.add_argument(
        "--edge-factor",
        type=float,
        default=1.5,
        help="Number of edges = round(edge_factor * number of nodes).",
    )
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.edge_factor < 1.0:
        raise SystemExit("--edge-factor must be >= 1.0")
    if len(set(args.sizes)) != len(args.sizes):
        raise SystemExit("--sizes contains duplicates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for index, n in enumerate(args.sizes):
        if n < 2:
            raise SystemExit(f"Invalid size: {n}")
        m = int(round(args.edge_factor * n))
        seed = args.base_seed + index * 100_003 + n
        name = f"scale_n{n}_m{m}"
        path = args.output_dir / f"{name}.json"

        if path.exists() and not args.overwrite:
            print(f"SKIP {path}")
        else:
            print(f"Generating {name} seed={seed} ...", flush=True)
            edges = generate_graph(n, m, seed)
            write_node_link_json(
                path,
                n=n,
                edges=edges,
                generator_seed=seed,
                edge_factor=args.edge_factor,
            )
            print(f"OK   {name}: N={n:,} M={m:,} -> {path}")

        manifest_rows.append(
            {
                "name": name,
                "num_nodes": n,
                "num_edges": m,
                "seed": seed,
                "file": str(path),
            }
        )

    index_path = args.output_dir / "scalability_graphs.meta"
    index_path.write_text(
        json.dumps(
            {
                "base_seed": args.base_seed,
                "edge_factor": args.edge_factor,
                "graphs": manifest_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
