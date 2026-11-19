#!/usr/bin/env python3
"""Build a fixed-size Qwen + Out-Degree hybrid candidate pool."""

from __future__ import annotations
import argparse, json, math, shutil, sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from rr_tools import load_processed_graph
from verified_search_fixed_budget import resolve_graph_path

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def dedup(nodes, n):
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

def hybrid_ranking(qwen, structural, pool_size):
    out, origin, seen = [], {}, set()
    qi = si = 0
    while len(out) < pool_size and (qi < len(qwen) or si < len(structural)):
        if qi < len(qwen):
            node = qwen[qi]; qi += 1
            if node not in seen:
                seen.add(node); out.append(node); origin[node] = "qwen"
                if len(out) >= pool_size:
                    break
        if si < len(structural):
            node = structural[si]; si += 1
            if node not in seen:
                seen.add(node); out.append(node); origin[node] = "out_degree"
                if len(out) >= pool_size:
                    break
    return out, origin

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--pool-size", type=int, default=60)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--clean-output", action="store_true")
    a = p.parse_args()

    input_root = a.input_root.resolve()
    output_root = a.output_root.resolve()
    if a.clean_output and output_root.exists():
        shutil.rmtree(output_root)

    paths = sorted(input_root.rglob("aggregate.json"))
    if not paths:
        raise RuntimeError(f"No aggregate.json under {input_root}")

    completed, failures = 0, []
    for aggregate_path in paths:
        try:
            aggregate = load_json(aggregate_path)
            graph_path = resolve_graph_path(str(aggregate["graph_file"]), input_root)
            _, graph = load_processed_graph(graph_path)
            n = graph.num_nodes

            qwen = dedup(aggregate.get("candidate_pool_ids", []), n)
            structural = sorted(
                range(1, n + 1),
                key=lambda node: (-graph.out_degree[node], -graph.in_degree[node], node),
            )
            pool, origin = hybrid_ranking(qwen, structural, min(a.pool_size, n))
            budgets = [int(x) for x in aggregate.get("budgets", [10, 20, 30])]
            if len(pool) < max(budgets):
                raise RuntimeError(f"Pool size {len(pool)} < max budget {max(budgets)}")

            rows = []
            for rank, node in enumerate(pool, start=1):
                rows.append({
                    "internal_node": node,
                    "original_node": node,
                    "consensus_score": 1.0 / math.log2(rank + 1.0),
                    "source_rank": rank,
                    "candidate_source": "hybrid_qwen_outdegree",
                    "hybrid_origin": origin[node],
                })

            variant = dict(aggregate)
            variant.update({
                "candidate_pool_size": len(pool),
                "candidate_pool_ids": pool,
                "candidate_pool": rows,
                "consensus_seed_baselines": {str(k): pool[:k] for k in budgets},
                "backend": "candidate_source_ablation",
                "model_name": "hybrid_qwen_outdegree",
                "ablation": {
                    "candidate_source": "hybrid_qwen_outdegree",
                    "pool_size": len(pool),
                    "requested_pool_size": a.pool_size,
                    "ranking_policy": "alternating_unique_qwen_outdegree",
                    "qwen_origin_count": sum(origin[n] == "qwen" for n in pool),
                    "outdegree_origin_count": sum(origin[n] == "out_degree" for n in pool),
                    "original_aggregate_file": str(aggregate_path),
                },
            })

            destination = output_root / "hybrid_qwen_outdegree" / aggregate_path.relative_to(input_root)
            if destination.exists() and not a.overwrite:
                completed += 1
                continue
            save_json(variant, destination)
            completed += 1
        except Exception as exc:
            failures.append({"aggregate": str(aggregate_path), "error": repr(exc)})

    save_json(
        {"completed": completed, "failures": failures, "pool_size": a.pool_size},
        output_root / "hybrid_manifest.json",
    )
    print(f"Completed: {completed}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {output_root / 'hybrid_qwen_outdegree'}")
    return 0 if not failures else 2

if __name__ == "__main__":
    raise SystemExit(main())
