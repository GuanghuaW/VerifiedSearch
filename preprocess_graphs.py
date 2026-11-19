#!/usr/bin/env python3
"""
Normalize synthetic and real-world graph files for influence-maximization experiments.

Supported input formats
-----------------------
Synthetic directory:
  - NetworkX node-link JSON files with `nodes` and `links` (or `edges`)

Real-world directory:
  - .edges / .edge / .txt / .csv edge lists
  - Matrix Market .mtx files

Outputs
-------
processed_graphs/
  synthetic/<graph_name>/graph.json
  synthetic/<graph_name>/adjacency.txt
  synthetic/<graph_name>/permutations/perm_000.json
  synthetic/<graph_name>/permutations/perm_000.txt
  real/<graph_name>/...
  manifest.json

The normalized graph uses contiguous internal node IDs 1..N while preserving a
mapping back to the original node labels. Duplicate edges and self-loops are
removed by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SUPPORTED_SYNTHETIC = {".json"}
SUPPORTED_REAL = {".edges", ".edge", ".txt", ".csv", ".mtx"}


@dataclass
class RawGraph:
    name: str
    dataset_type: str
    source_file: str
    directed: bool
    nodes: set[Any] = field(default_factory=set)
    edges: list[tuple[Any, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def natural_key(value: Any) -> tuple[int, Any]:
    """Sort numeric node labels numerically and other labels lexicographically."""
    if isinstance(value, bool):
        return (2, str(value))
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, float) and value.is_integer():
        return (0, int(value))
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return (0, int(text))
    return (1, text)


def parse_node_token(token: str) -> Any:
    token = token.strip()
    if re.fullmatch(r"[+-]?\d+", token):
        return int(token)
    try:
        value = float(token)
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        return token


def split_edge_line(line: str) -> list[str]:
    """Split whitespace-, comma-, or semicolon-separated edge-list rows."""
    return [part for part in re.split(r"[\s,;]+", line.strip()) if part]


def read_node_link_json(path: Path, dataset_type: str) -> RawGraph:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON object must be a dictionary.")

    graph = RawGraph(
        name=path.stem,
        dataset_type=dataset_type,
        source_file=str(path),
        directed=bool(data.get("directed", True)),
    )

    raw_nodes = data.get("nodes", [])
    for item in raw_nodes:
        if isinstance(item, dict):
            if "id" not in item:
                graph.warnings.append("A node object without an 'id' field was skipped.")
                continue
            graph.nodes.add(item["id"])
        else:
            graph.nodes.add(item)

    raw_edges = data.get("links", data.get("edges", []))
    for item in raw_edges:
        if isinstance(item, dict):
            if "source" not in item or "target" not in item:
                graph.warnings.append("An edge object without source/target was skipped.")
                continue
            source, target = item["source"], item["target"]
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            source, target = item[0], item[1]
        else:
            graph.warnings.append("An unrecognized edge entry was skipped.")
            continue
        graph.nodes.update((source, target))
        graph.edges.append((source, target))

    graph.metadata["input_directed"] = bool(data.get("directed", True))
    graph.metadata["input_multigraph"] = bool(data.get("multigraph", False))
    return graph


def read_edge_list(path: Path, dataset_type: str) -> RawGraph:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "%", "//")):
                continue
            parts = split_edge_line(stripped)
            if len(parts) < 2:
                continue
            rows.append(parts)

    if not rows:
        raise ValueError("No edge rows were found.")

    graph = RawGraph(
        name=path.stem,
        dataset_type=dataset_type,
        source_file=str(path),
        directed=True,
    )

    start = 0
    header_n: int | None = None
    header_m: int | None = None

    # Common Network Repository format:
    # first line = "<number_of_nodes> <number_of_edges>".
    # We only treat it as a header when the remaining row count matches m.
    try:
        first_a, first_b = int(rows[0][0]), int(rows[0][1])
        if len(rows) - 1 == first_b:
            header_n, header_m = first_a, first_b
            start = 1
            graph.metadata["detected_header"] = {
                "declared_nodes": header_n,
                "declared_edges": header_m,
            }
    except ValueError:
        pass

    for parts in rows[start:]:
        source = parse_node_token(parts[0])
        target = parse_node_token(parts[1])
        graph.nodes.update((source, target))
        graph.edges.append((source, target))

    if header_n is not None:
        numeric_nodes = {node for node in graph.nodes if isinstance(node, int)}
        if numeric_nodes and min(numeric_nodes) >= 1 and max(numeric_nodes) <= header_n:
            # Preserve isolated nodes declared in a reliable 1..N header.
            graph.nodes.update(range(1, header_n + 1))
        observed = len(graph.nodes)
        if observed != header_n:
            graph.warnings.append(
                f"Header declares {header_n} nodes, but {observed} node labels were "
                "observed after parsing. The observed node set is used."
            )
    return graph


def read_matrix_market(path: Path, dataset_type: str) -> RawGraph:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    if not lines:
        raise ValueError("Empty Matrix Market file.")

    header = lines[0].strip().lower()
    is_matrix_market = header.startswith("%%matrixmarket")
    is_symmetric = " symmetric" in header or " hermitian" in header

    data_lines = [
        line.strip()
        for line in lines[1 if is_matrix_market else 0 :]
        if line.strip() and not line.lstrip().startswith("%")
    ]
    if not data_lines:
        raise ValueError("No Matrix Market data rows were found.")

    dimensions = split_edge_line(data_lines[0])
    if len(dimensions) < 3:
        raise ValueError("Matrix Market dimension row must contain rows, columns, nnz.")

    n_rows, n_cols, declared_nnz = map(int, dimensions[:3])
    graph = RawGraph(
        name=path.stem,
        dataset_type=dataset_type,
        source_file=str(path),
        directed=not is_symmetric,
    )
    graph.nodes.update(range(1, max(n_rows, n_cols) + 1))
    graph.metadata["matrix_market"] = {
        "rows": n_rows,
        "columns": n_cols,
        "declared_nnz": declared_nnz,
        "symmetric": is_symmetric,
    }

    parsed_nnz = 0
    for line in data_lines[1:]:
        parts = split_edge_line(line)
        if len(parts) < 2:
            continue
        source = parse_node_token(parts[0])
        target = parse_node_token(parts[1])
        graph.edges.append((source, target))
        graph.nodes.update((source, target))
        parsed_nnz += 1
        if is_symmetric and source != target:
            graph.edges.append((target, source))

    if parsed_nnz != declared_nnz:
        graph.warnings.append(
            f"Matrix Market declares {declared_nnz} entries, but {parsed_nnz} were parsed."
        )
    if is_symmetric:
        graph.warnings.append(
            "Symmetric Matrix Market input was converted to a bidirectional directed graph."
        )
        graph.directed = True
    return graph


def load_graph(path: Path, dataset_type: str) -> RawGraph:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_node_link_json(path, dataset_type)
    if suffix == ".mtx":
        return read_matrix_market(path, dataset_type)
    return read_edge_list(path, dataset_type)


def normalize_graph(
    raw: RawGraph,
    make_bidirectional: bool = False,
    remove_self_loops: bool = True,
) -> dict[str, Any]:
    original_nodes = sorted(raw.nodes, key=natural_key)
    original_to_internal = {node: index + 1 for index, node in enumerate(original_nodes)}

    normalized_edges: set[tuple[int, int]] = set()
    removed_self_loops = 0
    duplicate_edges = 0

    for source, target in raw.edges:
        if source not in original_to_internal:
            original_to_internal[source] = len(original_to_internal) + 1
            original_nodes.append(source)
        if target not in original_to_internal:
            original_to_internal[target] = len(original_to_internal) + 1
            original_nodes.append(target)

        u = original_to_internal[source]
        v = original_to_internal[target]
        if remove_self_loops and u == v:
            removed_self_loops += 1
            continue
        if (u, v) in normalized_edges:
            duplicate_edges += 1
        normalized_edges.add((u, v))
        if make_bidirectional and u != v:
            normalized_edges.add((v, u))

    edges = sorted(normalized_edges)
    n = len(original_nodes)
    m = len(edges)

    adjacency: list[list[int]] = [[] for _ in range(n + 1)]
    reverse_adjacency: list[list[int]] = [[] for _ in range(n + 1)]
    for u, v in edges:
        adjacency[u].append(v)
        reverse_adjacency[v].append(u)

    for node in range(1, n + 1):
        adjacency[node].sort()
        reverse_adjacency[node].sort()

    out_degrees = [len(adjacency[node]) for node in range(1, n + 1)]
    in_degrees = [len(reverse_adjacency[node]) for node in range(1, n + 1)]
    isolated = [
        node
        for node in range(1, n + 1)
        if not adjacency[node] and not reverse_adjacency[node]
    ]

    density = m / (n * (n - 1)) if n > 1 else 0.0
    edge_digest = hashlib.sha256(
        "\n".join(f"{u} {v}" for u, v in edges).encode("utf-8")
    ).hexdigest()

    id_map = [
        {"internal": internal_id, "original": original}
        for internal_id, original in enumerate(original_nodes, start=1)
    ]

    return {
        "name": raw.name,
        "dataset_type": raw.dataset_type,
        "source_file": raw.source_file,
        "directed": True,
        "num_nodes": n,
        "num_edges": m,
        "nodes": list(range(1, n + 1)),
        "edges": [[u, v] for u, v in edges],
        "adjacency": {str(node): adjacency[node] for node in range(1, n + 1)},
        "reverse_adjacency": {
            str(node): reverse_adjacency[node] for node in range(1, n + 1)
        },
        "id_map": id_map,
        "stats": {
            "density": density,
            "average_out_degree": (sum(out_degrees) / n) if n else 0.0,
            "average_in_degree": (sum(in_degrees) / n) if n else 0.0,
            "max_out_degree": max(out_degrees, default=0),
            "max_in_degree": max(in_degrees, default=0),
            "isolated_nodes": len(isolated),
            "removed_self_loops": removed_self_loops,
            "removed_duplicate_edges": duplicate_edges,
            "sha256_edges": edge_digest,
        },
        "warnings": raw.warnings,
        "metadata": {
            **raw.metadata,
            "input_directed": raw.directed,
            "made_bidirectional": make_bidirectional,
        },
    }


def serialize_adjacency(
    graph: dict[str, Any],
    edges: Sequence[Sequence[int]] | None = None,
    title: str | None = None,
) -> str:
    n = int(graph["num_nodes"])
    selected_edges = graph["edges"] if edges is None else edges
    adjacency: list[list[int]] = [[] for _ in range(n + 1)]
    for u, v in selected_edges:
        adjacency[int(u)].append(int(v))
    for node in range(1, n + 1):
        adjacency[node].sort()

    header = title or (
        f"Directed graph {graph['name']} with {graph['num_nodes']} nodes "
        f"and {len(selected_edges)} edges."
    )
    lines = [header]
    lines.extend(
        f"Node {node}: [{', '.join(map(str, adjacency[node]))}]"
        for node in range(1, n + 1)
    )
    return "\n".join(lines) + "\n"


def make_permutation(
    graph: dict[str, Any],
    permutation_index: int,
    seed: int,
) -> dict[str, Any]:
    n = int(graph["num_nodes"])
    rng = random.Random(seed + permutation_index)
    shuffled = list(range(1, n + 1))
    rng.shuffle(shuffled)

    # original internal node i maps to shuffled[i - 1].
    internal_to_permuted = [0] + shuffled
    permuted_to_internal = [0] * (n + 1)
    for internal, permuted in enumerate(shuffled, start=1):
        permuted_to_internal[permuted] = internal

    permuted_edges = sorted(
        [
            [internal_to_permuted[int(u)], internal_to_permuted[int(v)]]
            for u, v in graph["edges"]
        ]
    )

    return {
        "graph_name": graph["name"],
        "permutation_index": permutation_index,
        "permutation_seed": seed + permutation_index,
        "num_nodes": n,
        "num_edges": len(permuted_edges),
        "directed": True,
        "internal_to_permuted": internal_to_permuted[1:],
        "permuted_to_internal": permuted_to_internal[1:],
        "edges": permuted_edges,
    }


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def discover_files(directory: Path, allowed_suffixes: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed_suffixes
        and not any(part.lower() in {"bak", "backup", "processed_graphs"} for part in path.parts)
    )


def process_one(
    source_path: Path,
    dataset_type: str,
    output_root: Path,
    num_permutations: int,
    permutation_seed: int,
    make_bidirectional: bool,
) -> dict[str, Any]:
    raw = load_graph(source_path, dataset_type)
    graph = normalize_graph(
        raw,
        make_bidirectional=make_bidirectional,
        remove_self_loops=True,
    )

    graph_dir = output_root / dataset_type / graph["name"]
    save_json(graph, graph_dir / "graph.json")
    (graph_dir / "adjacency.txt").write_text(
        serialize_adjacency(graph), encoding="utf-8"
    )

    permutation_files: list[str] = []
    for index in range(num_permutations):
        permutation = make_permutation(graph, index, permutation_seed)
        json_path = graph_dir / "permutations" / f"perm_{index:03d}.json"
        text_path = graph_dir / "permutations" / f"perm_{index:03d}.txt"
        save_json(permutation, json_path)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(
            serialize_adjacency(
                graph,
                edges=permutation["edges"],
                title=(
                    f"Directed permuted graph {graph['name']} "
                    f"(permutation {index}) with {graph['num_nodes']} nodes "
                    f"and {graph['num_edges']} edges."
                ),
            ),
            encoding="utf-8",
        )
        permutation_files.append(str(json_path))

    return {
        "name": graph["name"],
        "dataset_type": dataset_type,
        "source_file": str(source_path),
        "processed_file": str(graph_dir / "graph.json"),
        "adjacency_file": str(graph_dir / "adjacency.txt"),
        "num_nodes": graph["num_nodes"],
        "num_edges": graph["num_edges"],
        "density": graph["stats"]["density"],
        "isolated_nodes": graph["stats"]["isolated_nodes"],
        "warnings": graph["warnings"],
        "permutation_files": permutation_files,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize graph files for influence-maximization experiments."
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=Path("sdata"),
        help="Directory containing synthetic NetworkX JSON graphs.",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=Path("ndata"),
        help="Directory containing real-world edge-list or Matrix Market graphs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_graphs"),
        help="Output directory.",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=5,
        help="Number of node-ID permutations generated for each graph.",
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=2026,
        help="Base random seed for node permutations.",
    )
    parser.add_argument(
        "--real-bidirectional",
        action="store_true",
        help=(
            "Convert every real-world edge u->v into both u->v and v->u. "
            "Do not enable this when real-world files are already directed."
        ),
    )
    parser.add_argument(
        "--synthetic-bidirectional",
        action="store_true",
        help="Convert synthetic edges to bidirectional edges.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    if args.num_permutations < 0:
        print("ERROR: --num-permutations must be nonnegative.", file=sys.stderr)
        return 2

    synthetic_files = discover_files(args.synthetic_dir, SUPPORTED_SYNTHETIC)
    real_files = discover_files(args.real_dir, SUPPORTED_REAL)

    if not synthetic_files and not real_files:
        print(
            "No supported graph files found. Check --synthetic-dir and --real-dir.",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    jobs = [
        *[(path, "synthetic", args.synthetic_bidirectional) for path in synthetic_files],
        *[(path, "real", args.real_bidirectional) for path in real_files],
    ]

    for path, dataset_type, bidirectional in jobs:
        try:
            entry = process_one(
                source_path=path,
                dataset_type=dataset_type,
                output_root=args.output_dir,
                num_permutations=args.num_permutations,
                permutation_seed=args.permutation_seed,
                make_bidirectional=bidirectional,
            )
            entries.append(entry)
            warning_marker = " [warning]" if entry["warnings"] else ""
            print(
                f"OK  {dataset_type:9s} {entry['name']:<35s} "
                f"N={entry['num_nodes']:<6d} M={entry['num_edges']:<7d}"
                f"{warning_marker}"
            )
            for warning in entry["warnings"]:
                print(f"    - {warning}")
        except Exception as exc:
            failures.append({"source_file": str(path), "error": str(exc)})
            print(f"FAIL {path}: {exc}", file=sys.stderr)

    manifest = {
        "synthetic_dir": str(args.synthetic_dir),
        "real_dir": str(args.real_dir),
        "output_dir": str(args.output_dir),
        "num_permutations": args.num_permutations,
        "permutation_seed": args.permutation_seed,
        "graphs": entries,
        "failures": failures,
    }
    save_json(manifest, args.output_dir / "manifest.json")

    print(
        f"\nProcessed {len(entries)} graph(s); {len(failures)} failure(s). "
        f"Manifest: {args.output_dir / 'manifest.json'}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
