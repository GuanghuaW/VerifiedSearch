#!/usr/bin/env python3
"""Build budget-isolated WDD refinement candidate-control inputs.

For each graph/diffusion/budget, the pool contains:
  exact WDD seed set for that budget
  + replacement candidates from one controlled source.

Sources:
  qwen       : Qwen multi-permutation extras
  out_degree : directed out-degree extras
  hybrid     : alternating Qwen/out-degree extras

Each output aggregate contains exactly one budget and a fixed total pool size.
Compatible with verified_search_ablation.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rr_tools import load_processed_graph
from verified_search_fixed_budget import resolve_graph_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def parse_seed_set(value: Any) -> list[int]:
    if isinstance(value, list):
        raw = value
    else:
        text = str(value).strip()
        try:
            raw = json.loads(text)
        except Exception:
            raw = ast.literal_eval(text)
    out, seen = [], set()
    for item in raw:
        node = int(item)
        if node not in seen:
            seen.add(node)
            out.append(node)
    return out


def dedup_valid(nodes: list[Any], n: int) -> list[int]:
    out, seen = [], set()
    for raw in nodes:
        try:
            node = int(raw)
        except Exception:
            continue
        if 1 <= node <= n and node not in seen:
            seen.add(node)
            out.append(node)
    return out


def build_extras(source, initial, qwen, outdeg, needed):
    seen = set(initial)
    extras = []
    origin = {}

    def add_one(ranking, label, cursor):
        while cursor < len(ranking):
            node = ranking[cursor]
            cursor += 1
            if node in seen:
                continue
            seen.add(node)
            extras.append(node)
            origin[node] = label
            return cursor, True
        return cursor, False

    if source in ("qwen", "out_degree"):
        ranking = qwen if source == "qwen" else outdeg
        label = source
        cursor = 0
        while len(extras) < needed:
            cursor, added = add_one(ranking, label, cursor)
            if not added:
                break

    elif source == "hybrid":
        qi = oi = 0
        turn_qwen = True
        while len(extras) < needed:
            if turn_qwen:
                qi, added = add_one(qwen, "qwen", qi)
                if not added:
                    oi, added = add_one(outdeg, "out_degree", oi)
            else:
                oi, added = add_one(outdeg, "out_degree", oi)
                if not added:
                    qi, added = add_one(qwen, "qwen", qi)
            if not added:
                break
            turn_qwen = not turn_qwen
    else:
        raise ValueError(source)

    if len(extras) < needed:
        raise RuntimeError(
            f"{source}: only {len(extras)} extras available; need {needed}"
        )
    return extras[:needed], origin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--wdd-summary", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--sources",
        nargs="+",
        choices=["qwen", "out_degree", "hybrid"],
        default=["qwen", "out_degree", "hybrid"],
    )
    p.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--pool-size", type=int, default=60)
    p.add_argument("--clean-output", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    input_root = a.input_root.expanduser().resolve()
    output_root = a.output_root.expanduser().resolve()

    if a.clean_output and output_root.exists():
        shutil.rmtree(output_root)

    wdd = pd.read_csv(a.wdd_summary)
    required = {
        "graph_name", "dataset_type", "diffusion", "budget", "seed_set"
    }
    missing = required - set(wdd.columns)
    if missing:
        raise ValueError(
            f"WDD summary missing {sorted(missing)}; columns={list(wdd.columns)}"
        )

    seed_map = {}
    for _, row in wdd.iterrows():
        key = (
            str(row["graph_name"]),
            str(row["dataset_type"]),
            str(row["diffusion"]),
            int(row["budget"]),
        )
        seed_map[key] = parse_seed_set(row["seed_set"])

    aggregates = sorted(input_root.rglob("aggregate.json"))
    if not aggregates:
        raise RuntimeError(f"No aggregate.json under {input_root}")

    counts = {}
    failures = []

    for aggregate_path in aggregates:
        try:
            aggregate = load_json(aggregate_path)
            graph_name = str(aggregate["graph_name"])
            dataset_type = str(aggregate["dataset_type"])
            diffusion = str(aggregate["diffusion"])

            graph_path = resolve_graph_path(
                str(aggregate["graph_file"]),
                input_root,
            )
            _, graph = load_processed_graph(graph_path)
            n = graph.num_nodes

            qwen = dedup_valid(
                aggregate.get("candidate_pool_ids", []),
                n,
            )
            outdeg = sorted(
                range(1, n + 1),
                key=lambda node: (
                    -graph.out_degree[node],
                    -graph.in_degree[node],
                    node,
                ),
            )

            for budget in a.budgets:
                key = (
                    graph_name,
                    dataset_type,
                    diffusion,
                    int(budget),
                )
                if key not in seed_map:
                    raise KeyError(f"Missing WDD seeds for {key}")

                initial = seed_map[key]
                if len(initial) != budget:
                    raise ValueError(
                        f"{key}: expected {budget} seeds, got {len(initial)}"
                    )

                target_pool = min(a.pool_size, n)
                if target_pool < budget:
                    raise ValueError(
                        f"pool_size={target_pool} < budget={budget}"
                    )
                needed = target_pool - budget

                for source in a.sources:
                    extras, origin = build_extras(
                        source, initial, qwen, outdeg, needed
                    )
                    pool = initial + extras

                    rows = []
                    for rank, node in enumerate(pool, start=1):
                        pool_origin = (
                            "wdd_initialization"
                            if rank <= budget
                            else origin[node]
                        )
                        rows.append(
                            {
                                "internal_node": node,
                                "original_node": node,
                                "consensus_score": 1.0 / math.log2(rank + 1.0),
                                "source_rank": rank,
                                "candidate_source": f"wdd_init_{source}",
                                "pool_origin": pool_origin,
                            }
                        )

                    variant = dict(aggregate)
                    variant.update(
                        {
                            "budgets": [budget],
                            "candidate_pool_size": len(pool),
                            "candidate_pool_ids": pool,
                            "candidate_pool": rows,
                            "consensus_seed_baselines": {
                                str(budget): initial
                            },
                            "backend": (
                                "wdd_budget_isolated_candidate_control"
                            ),
                            "model_name": f"wdd_init_{source}",
                            "ablation": {
                                "candidate_source": source,
                                "initialization_source": (
                                    "weighted_degree_discount"
                                ),
                                "budget_isolated": True,
                                "budget": budget,
                                "pool_size": len(pool),
                                "replacement_slots": len(extras),
                                "replacement_qwen_count": sum(
                                    origin[node] == "qwen"
                                    for node in extras
                                ),
                                "replacement_outdegree_count": sum(
                                    origin[node] == "out_degree"
                                    for node in extras
                                ),
                            },
                        }
                    )

                    relative = aggregate_path.relative_to(input_root)
                    dest = (
                        output_root
                        / source
                        / f"budget_{budget}"
                        / relative
                    )
                    if not dest.exists() or a.overwrite:
                        save_json(variant, dest)

                    counts[(source, budget)] = counts.get(
                        (source, budget), 0
                    ) + 1

        except Exception as exc:
            failures.append(
                {"aggregate": str(aggregate_path), "error": repr(exc)}
            )

    manifest = {
        "input_root": str(input_root),
        "wdd_summary": str(a.wdd_summary),
        "pool_size": a.pool_size,
        "sources": a.sources,
        "budgets": a.budgets,
        "counts": {
            f"{s}_budget_{b}": counts.get((s, b), 0)
            for s in a.sources
            for b in a.budgets
        },
        "failures": failures,
    }
    save_json(
        manifest,
        output_root / "candidate_control_manifest.json",
    )

    for s in a.sources:
        for b in a.budgets:
            print(f"{s:10s} budget={b:2d}: {counts.get((s,b),0)}")
    print("Failures:", len(failures))
    print("Manifest:", output_root / "candidate_control_manifest.json")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
