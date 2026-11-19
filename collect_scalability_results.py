#!/usr/bin/env python3
"""Collect scalability outputs into paper-ready CSV tables.

Missing methods are tolerated so the collector can be run incrementally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_float(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def load_method_summary(
    path: Path,
    *,
    forced_method: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        method = (
            forced_method
            or raw.get("method")
            or raw.get("controller")
            or path.stem.removeprefix("summary_")
        )
        rows.append(
            {
                "method": str(method),
                "graph_name": raw.get("graph_name"),
                "diffusion": raw.get("diffusion"),
                "budget": int(float(raw.get("budget", 0))),
                "num_nodes": (
                    int(float(raw["num_nodes"]))
                    if raw.get("num_nodes")
                    else None
                ),
                "num_edges": (
                    int(float(raw["num_edges"]))
                    if raw.get("num_edges")
                    else None
                ),
                "final_eval_spread": first_float(
                    raw, "final_eval_spread", "final_spread"
                ),
                "normalized_final_spread": first_float(
                    raw, "normalized_final_spread", "normalized_spread"
                ),
                "runtime_seconds": first_float(
                    raw,
                    "runtime_seconds",
                    "selection_seconds",
                    "elapsed_seconds",
                ),
                "accepted_swaps": first_float(raw, "accepted_swaps"),
                "verified_swaps": first_float(raw, "verified_swaps"),
                "result_file": raw.get("result_file", ""),
            }
        )
    return rows


def candidate_cost_rows(candidate_root: Path) -> list[dict[str, Any]]:
    rows = []
    for aggregate_path in sorted(candidate_root.rglob("aggregate.json")):
        aggregate = json.loads(
            aggregate_path.read_text(encoding="utf-8")
        )
        run_paths = sorted((aggregate_path.parent / "runs").glob("perm_*.json"))

        input_tokens: list[int] = []
        output_tokens: list[int] = []
        generation_seconds: list[float] = []
        wall_seconds: list[float] = []
        valid_runs = 0
        attempts_total = 0

        for run_path in run_paths:
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if run.get("status") == "ok":
                valid_runs += 1
            for attempt in run.get("attempts", []):
                attempts_total += 1
                meta = (
                    attempt.get("backend_metadata")
                    or attempt.get("generation_metadata")
                    or {}
                )
                if meta.get("input_tokens") is not None:
                    input_tokens.append(int(meta["input_tokens"]))
                if meta.get("output_tokens") is not None:
                    output_tokens.append(int(meta["output_tokens"]))
                if meta.get("generation_seconds") is not None:
                    generation_seconds.append(
                        float(meta["generation_seconds"])
                    )
                if attempt.get("wall_seconds") is not None:
                    wall_seconds.append(float(attempt["wall_seconds"]))

        stability = aggregate.get("permutation_stability", {})
        rows.append(
            {
                "graph_name": str(aggregate["graph_name"]),
                "diffusion": aggregate.get("diffusion"),
                "num_nodes": int(aggregate.get("num_nodes", 0)),
                "num_edges": int(aggregate.get("num_edges", 0)),
                "candidate_pool_size": int(
                    aggregate.get("candidate_pool_size", 0)
                ),
                "num_permutations": len(run_paths),
                "valid_permutations": valid_runs,
                "generation_attempts": attempts_total,
                "mean_input_tokens": (
                    statistics.fmean(input_tokens)
                    if input_tokens else None
                ),
                "max_input_tokens": (
                    max(input_tokens) if input_tokens else None
                ),
                "mean_output_tokens": (
                    statistics.fmean(output_tokens)
                    if output_tokens else None
                ),
                "candidate_generation_seconds_total": (
                    sum(generation_seconds)
                    if generation_seconds else None
                ),
                "candidate_generation_seconds_mean_attempt": (
                    statistics.fmean(generation_seconds)
                    if generation_seconds else None
                ),
                "candidate_wall_seconds_total": (
                    sum(wall_seconds) if wall_seconds else None
                ),
                "top30_mean_pairwise_jaccard": stability.get(
                    "top_30_mean_pairwise_jaccard"
                ),
                "aggregate_file": str(aggregate_path),
            }
        )
    return rows


def controller_llm_seconds(result_file: str, budget: int) -> float | None:
    if not result_file:
        return None
    path = Path(result_file)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        item = payload.get("budgets", {}).get(str(budget), {})
        history = (
            item.get("repair_history")
            or item.get("history")
            or item.get("verified_search", {}).get("repair_history")
            or []
        )
        total = 0.0
        found = False
        for batch in history:
            for attempt in batch.get("llm_controller_attempts", []):
                meta = (
                    attempt.get("backend_metadata")
                    or attempt.get("generation_metadata")
                    or {}
                )
                if meta.get("generation_seconds") is not None:
                    total += float(meta["generation_seconds"])
                    found = True
        return total if found else None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("VerifiedSearch/scalability"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("VerifiedSearch/scalability/analysis"),
    )
    args = parser.parse_args()

    root = args.root
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    candidate_root = root / "candidate_outputs_qwen35_9b"
    candidate_rows = candidate_cost_rows(candidate_root)

    method_specs = [
        (root / "internal" / "summary_qwen_multi_direct.csv", None),
        (root / "internal" / "summary_out_degree.csv", None),
        (root / "internal" / "summary_weighted_out_degree.csv", None),
        (
            root
            / "internal"
            / "summary_weighted_degree_discount.csv",
            None,
        ),
        (root / "internal" / "summary_rr_greedy.csv", None),
        (root / "imm" / "summary_imm_w1.csv", None),
        (
            root / "verified_proxy" / "summary_proxy_fixed.csv",
            "proxy_fixed",
        ),
        (
            root
            / "verified_llm"
            / "summary_llm_fixed_randomfill.csv",
            "llm_fixed_randomfill",
        ),
    ]

    method_rows: list[dict[str, Any]] = []
    for path, forced in method_specs:
        method_rows.extend(
            load_method_summary(path, forced_method=forced)
        )

    graph_meta = {row["graph_name"]: row for row in candidate_rows}
    for row in method_rows:
        meta = graph_meta.get(str(row["graph_name"]))
        if meta:
            row["num_nodes"] = row["num_nodes"] or meta["num_nodes"]
            row["num_edges"] = row["num_edges"] or meta["num_edges"]
        if row["method"].startswith("llm_fixed"):
            row["controller_llm_generation_seconds"] = (
                controller_llm_seconds(
                    row.get("result_file", ""),
                    row["budget"],
                )
            )

    if candidate_rows:
        candidate_path = output / "scalability_candidate_cost.csv"
        with candidate_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(candidate_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(candidate_rows)

    if method_rows:
        method_rows.sort(
            key=lambda row: (
                row.get("num_nodes") or math.inf,
                str(row["method"]),
            )
        )
        long_path = output / "scalability_methods_long.csv"
        fields = []
        for row in method_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with long_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(method_rows)

    wide_rows = []
    methods = sorted({row["method"] for row in method_rows})
    for candidate in sorted(
        candidate_rows,
        key=lambda row: row["num_nodes"],
    ):
        graph_name = candidate["graph_name"]
        wide: dict[str, Any] = {
            "graph_name": graph_name,
            "num_nodes": candidate["num_nodes"],
            "num_edges": candidate["num_edges"],
            "mean_input_tokens": candidate["mean_input_tokens"],
            "max_input_tokens": candidate["max_input_tokens"],
            "candidate_generation_seconds_total": candidate[
                "candidate_generation_seconds_total"
            ],
            "valid_permutations": candidate["valid_permutations"],
            "candidate_pool_size": candidate["candidate_pool_size"],
            "top30_mean_pairwise_jaccard": candidate[
                "top30_mean_pairwise_jaccard"
            ],
        }
        for method in methods:
            matches = [
                row
                for row in method_rows
                if row["graph_name"] == graph_name
                and row["method"] == method
                and row["budget"] == 30
                and row["diffusion"] == "weighted_cascade"
            ]
            if not matches:
                continue
            row = matches[0]
            wide[f"{method}__spread"] = row["final_eval_spread"]
            wide[f"{method}__norm"] = row[
                "normalized_final_spread"
            ]
            wide[f"{method}__runtime"] = row["runtime_seconds"]
            if (
                row.get("controller_llm_generation_seconds")
                is not None
            ):
                wide[
                    f"{method}__controller_llm_seconds"
                ] = row[
                    "controller_llm_generation_seconds"
                ]
        wide_rows.append(wide)

    if wide_rows:
        wide_path = output / "scalability_paper_table.csv"
        fields = []
        for row in wide_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with wide_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(wide_rows)

    print(f"Candidate rows: {len(candidate_rows)}")
    print(f"Method rows:    {len(method_rows)}")
    print(f"Output:         {output}")
    for path in sorted(output.glob("*.csv")):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
