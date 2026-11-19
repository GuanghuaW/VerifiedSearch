#!/usr/bin/env python3
"""Collect token/runtime/verification costs for the main benchmark."""

from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import pandas as pd

KEYS = ["graph_name", "dataset_type", "diffusion"]

def candidate_costs(root: Path):
    rows = []
    for aggregate_path in sorted(root.rglob("aggregate.json")):
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        ins, outs, gens, walls = [], [], [], []
        attempts = 0
        for run_path in sorted((aggregate_path.parent / "runs").glob("perm_*.json")):
            run = json.loads(run_path.read_text(encoding="utf-8"))
            for attempt in run.get("attempts", []):
                attempts += 1
                meta = attempt.get("backend_metadata") or attempt.get("generation_metadata") or {}
                if meta.get("input_tokens") is not None:
                    ins.append(int(meta["input_tokens"]))
                if meta.get("output_tokens") is not None:
                    outs.append(int(meta["output_tokens"]))
                if meta.get("generation_seconds") is not None:
                    gens.append(float(meta["generation_seconds"]))
                if attempt.get("wall_seconds") is not None:
                    walls.append(float(attempt["wall_seconds"]))
        rows.append({
            "graph_name": aggregate["graph_name"],
            "dataset_type": aggregate["dataset_type"],
            "diffusion": aggregate["diffusion"],
            "candidate_attempts": attempts,
            "candidate_input_tokens_total": sum(ins) if ins else None,
            "candidate_input_tokens_mean": statistics.fmean(ins) if ins else None,
            "candidate_output_tokens_total": sum(outs) if outs else None,
            "candidate_generation_seconds_total": sum(gens) if gens else None,
            "candidate_wall_seconds_total": sum(walls) if walls else None,
        })
    return pd.DataFrame(rows)

def controller_cost(result_file, budget):
    path = Path(str(result_file))
    empty = {
        "controller_calls": None,
        "controller_input_tokens_total": None,
        "controller_output_tokens_total": None,
        "controller_generation_seconds_total": None,
    }
    if not path.exists():
        return empty
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload.get("budgets", {}).get(str(int(budget)), {})
    history = item.get("repair_history", [])
    calls = 0
    ins = outs = 0
    secs = 0.0
    token_found = time_found = False
    for batch in history:
        for attempt in batch.get("llm_controller_attempts", []):
            calls += 1
            meta = attempt.get("backend_metadata") or attempt.get("generation_metadata") or {}
            if meta.get("input_tokens") is not None:
                ins += int(meta["input_tokens"]); token_found = True
            if meta.get("output_tokens") is not None:
                outs += int(meta["output_tokens"]); token_found = True
            if meta.get("generation_seconds") is not None:
                secs += float(meta["generation_seconds"]); time_found = True
    return {
        "controller_calls": calls,
        "controller_input_tokens_total": ins if token_found else None,
        "controller_output_tokens_total": outs if token_found else None,
        "controller_generation_seconds_total": secs if time_found else None,
    }

def runtime_column(df):
    for name in ["runtime_seconds", "selection_seconds", "elapsed_seconds"]:
        if name in df.columns:
            return name
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--verified-summary", type=Path, required=True)
    p.add_argument("--proxy-summary", type=Path)
    p.add_argument("--direct-summary", type=Path)
    p.add_argument("--imm-summary", type=Path)
    p.add_argument("--rr-summary", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()

    cand = candidate_costs(a.candidate_root)
    verified = pd.read_csv(a.verified_summary)
    rows = []
    for _, row in verified.iterrows():
        item = {
            "method": "VerifiedSearch-LLM",
            "graph_name": row["graph_name"],
            "dataset_type": row["dataset_type"],
            "diffusion": row["diffusion"],
            "budget": int(row["budget"]),
            "verified_swaps": row.get("verified_swaps"),
            "accepted_swaps": row.get("accepted_swaps"),
            "final_eval_spread": row.get("final_eval_spread"),
        }
        item.update(controller_cost(row.get("result_file", ""), int(row["budget"])))
        rows.append(item)

    detailed = pd.DataFrame(rows).merge(cand, on=KEYS, how="left")
    detailed["llm_generation_seconds_total"] = (
        detailed["candidate_generation_seconds_total"].fillna(0)
        + detailed["controller_generation_seconds_total"].fillna(0)
    )
    detailed["llm_input_tokens_total"] = (
        detailed["candidate_input_tokens_total"].fillna(0)
        + detailed["controller_input_tokens_total"].fillna(0)
    )
    detailed["llm_output_tokens_total"] = (
        detailed["candidate_output_tokens_total"].fillna(0)
        + detailed["controller_output_tokens_total"].fillna(0)
    )

    a.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(a.output_dir / "verified_llm_cost_by_instance.csv", index=False)

    summary = detailed.groupby("method", as_index=False).agg(
        instances=("budget", "count"),
        mean_candidate_input_tokens=("candidate_input_tokens_total", "mean"),
        mean_controller_input_tokens=("controller_input_tokens_total", "mean"),
        mean_total_llm_input_tokens=("llm_input_tokens_total", "mean"),
        mean_candidate_generation_seconds=("candidate_generation_seconds_total", "mean"),
        mean_controller_generation_seconds=("controller_generation_seconds_total", "mean"),
        mean_total_llm_generation_seconds=("llm_generation_seconds_total", "mean"),
        mean_verified_swaps=("verified_swaps", "mean"),
    )
    summary.to_csv(a.output_dir / "main_cost_summary.csv", index=False)

    extras = []
    for label, path in [
        ("VerifiedSearch-Proxy", a.proxy_summary),
        ("Direct Qwen Multi", a.direct_summary),
        ("IMM-W1", a.imm_summary),
        ("RR-Greedy", a.rr_summary),
    ]:
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path).copy()
        df["paper_method"] = label
        rc = runtime_column(df)
        df["reported_runtime_seconds"] = pd.to_numeric(df[rc], errors="coerce") if rc else pd.NA
        extras.append(df)
    if extras:
        pd.concat(extras, ignore_index=True, sort=False).to_csv(
            a.output_dir / "other_methods_runtime_records.csv", index=False
        )

    print(summary.to_string(index=False))
    print(f"\nSaved to: {a.output_dir}")

if __name__ == "__main__":
    main()
