#!/usr/bin/env python3
"""Build aggregates for Weighted-Degree-Discount initialization + Qwen refinement."""

from __future__ import annotations
import argparse, ast, json, math, shutil
from pathlib import Path
from typing import Any
import pandas as pd

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def parse_seeds(value: Any):
    if isinstance(value, list):
        raw = value
    else:
        text = str(value).strip()
        try:
            raw = json.loads(text)
        except Exception:
            raw = ast.literal_eval(text)
    out, seen = [], set()
    for x in raw:
        node = int(x)
        if node not in seen:
            seen.add(node); out.append(node)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--wdd-summary", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--pool-size", type=int, default=60)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--clean-output", action="store_true")
    a = p.parse_args()

    input_root = a.input_root.resolve()
    output_root = a.output_root.resolve()
    if a.clean_output and output_root.exists():
        shutil.rmtree(output_root)

    wdd = pd.read_csv(a.wdd_summary)
    required = {"graph_name", "dataset_type", "diffusion", "budget", "seed_set"}
    missing = required - set(wdd.columns)
    if missing:
        raise ValueError(f"WDD summary missing {sorted(missing)}; columns={list(wdd.columns)}")

    seed_map = {}
    for _, row in wdd.iterrows():
        key = (
            str(row["graph_name"]),
            str(row["dataset_type"]),
            str(row["diffusion"]),
            int(row["budget"]),
        )
        seed_map[key] = parse_seeds(row["seed_set"])

    completed, failures = 0, []
    for aggregate_path in sorted(input_root.rglob("aggregate.json")):
        try:
            aggregate = load_json(aggregate_path)
            graph_name = str(aggregate["graph_name"])
            dataset_type = str(aggregate["dataset_type"])
            diffusion = str(aggregate["diffusion"])
            budgets = sorted(int(x) for x in aggregate.get("budgets", [10,20,30]))

            wdd_by_budget, reserve, seen = {}, [], set()
            for budget in budgets:
                key = (graph_name, dataset_type, diffusion, budget)
                if key not in seed_map:
                    raise KeyError(f"Missing WDD seeds for {key}")
                seeds = seed_map[key]
                if len(seeds) != budget:
                    raise ValueError(f"{key}: expected {budget} seeds, got {len(seeds)}")
                wdd_by_budget[str(budget)] = seeds
                for node in seeds:
                    if node not in seen:
                        seen.add(node); reserve.append(node)

            if len(reserve) > a.pool_size:
                raise ValueError(f"WDD seed union {len(reserve)} > pool size {a.pool_size}")

            qwen_extra = []
            for raw in aggregate.get("candidate_pool_ids", []):
                node = int(raw)
                if node not in seen:
                    seen.add(node); qwen_extra.append(node)

            pool = reserve + qwen_extra[: max(0, a.pool_size - len(reserve))]
            if len(pool) < max(budgets):
                raise ValueError("Candidate pool smaller than max budget")

            reserve_set = set(reserve)
            rows = []
            for rank, node in enumerate(pool, start=1):
                rows.append({
                    "internal_node": node,
                    "original_node": node,
                    "consensus_score": 1.0 / math.log2(rank + 1.0),
                    "source_rank": rank,
                    "candidate_source": "wdd_init_qwen_refinement",
                    "pool_origin": "wdd_initialization" if node in reserve_set else "qwen_multi",
                })

            variant = dict(aggregate)
            variant.update({
                "candidate_pool_size": len(pool),
                "candidate_pool_ids": pool,
                "candidate_pool": rows,
                "consensus_seed_baselines": wdd_by_budget,
                "backend": "strong_initialization_ablation",
                "model_name": "wdd_init_qwen_refinement",
                "ablation": {
                    "candidate_source": "wdd_init_qwen_refinement",
                    "initialization_source": "weighted_degree_discount",
                    "replacement_source": "qwen_multi_plus_wdd_union",
                    "pool_size": len(pool),
                    "requested_pool_size": a.pool_size,
                    "reserved_wdd_union": len(reserve),
                    "original_aggregate_file": str(aggregate_path),
                    "wdd_summary": str(a.wdd_summary),
                },
            })

            destination = output_root / aggregate_path.relative_to(input_root)
            if destination.exists() and not a.overwrite:
                completed += 1
                continue
            save_json(variant, destination)
            completed += 1
        except Exception as exc:
            failures.append({"aggregate": str(aggregate_path), "error": repr(exc)})

    save_json(
        {"completed": completed, "failures": failures, "pool_size": a.pool_size},
        output_root / "wdd_refinement_manifest.json",
    )
    print(f"Completed: {completed}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {output_root}")
    return 0 if not failures else 2

if __name__ == "__main__":
    raise SystemExit(main())
