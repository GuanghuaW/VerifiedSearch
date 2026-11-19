#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

KEYS = ["graph_name","dataset_type","diffusion","budget"]

def pick(df, names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"Missing one of {names}; columns={list(df.columns)}")
    return None

def add_norm(df, node_map):
    out = df.copy()
    if "num_nodes" not in out.columns:
        out = out.merge(node_map, on=KEYS, how="left", validate="one_to_one")
    s = pick(out, ["final_eval_spread","final_spread","mean_final_spread"])
    out["final_eval_spread"] = pd.to_numeric(out[s], errors="raise")
    if "normalized_final_spread" in out.columns:
        out["normalized_final_spread"] = pd.to_numeric(out["normalized_final_spread"], errors="coerce")
    else:
        out["normalized_final_spread"] = out["final_eval_spread"] / pd.to_numeric(out["num_nodes"], errors="raise")
    return out

def read_method(path, method, node_map):
    df = add_norm(pd.read_csv(path), node_map)
    df["paper_method"] = method
    rc = pick(df, ["runtime_seconds","selection_seconds","elapsed_seconds"], False)
    df["reported_runtime_seconds"] = pd.to_numeric(df[rc], errors="coerce") if rc else pd.NA
    keep = ["paper_method","graph_name","dataset_type","diffusion","budget","num_nodes",
            "final_eval_spread","normalized_final_spread","reported_runtime_seconds"]
    for c in ["verified_swaps","accepted_swaps","verification_budget","result_file"]:
        if c in df.columns:
            keep.append(c)
    return df[keep]

def candidate_costs(root):
    rows = []
    for ap in sorted(Path(root).rglob("aggregate.json")):
        agg = json.loads(ap.read_text(encoding="utf-8"))
        ins=[]; outs=[]; gens=[]; walls=[]
        for rp in sorted((ap.parent/"runs").glob("perm_*.json")):
            run = json.loads(rp.read_text(encoding="utf-8"))
            for att in run.get("attempts",[]):
                meta = att.get("backend_metadata") or att.get("generation_metadata") or {}
                if meta.get("input_tokens") is not None: ins.append(float(meta["input_tokens"]))
                if meta.get("output_tokens") is not None: outs.append(float(meta["output_tokens"]))
                if meta.get("generation_seconds") is not None: gens.append(float(meta["generation_seconds"]))
                if att.get("wall_seconds") is not None: walls.append(float(att["wall_seconds"]))
        rows.append({
            "graph_name": str(agg["graph_name"]),
            "dataset_type": str(agg["dataset_type"]),
            "diffusion": str(agg["diffusion"]),
            "candidate_input_tokens_total": sum(ins) if ins else None,
            "candidate_input_tokens_mean_per_permutation": sum(ins)/len(ins) if ins else None,
            "candidate_output_tokens_total": sum(outs) if outs else None,
            "candidate_generation_seconds_total": sum(gens) if gens else None,
            "candidate_wall_seconds_total": sum(walls) if walls else None,
            "candidate_llm_calls": len(ins),
        })
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--internal-dir", type=Path, required=True)
    p.add_argument("--imm-summary", type=Path, required=True)
    p.add_argument("--wdd-proxy-summary", type=Path, required=True)
    p.add_argument("--wdd-llm-summary", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--prompt-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a=p.parse_args()

    wdd_base = pd.read_csv(a.internal_dir/"summary_weighted_degree_discount.csv")
    if "num_nodes" not in wdd_base.columns:
        raise RuntimeError("WDD scalability summary must contain num_nodes")
    node_map = wdd_base[KEYS+["num_nodes"]].drop_duplicates()

    specs = [
        ("Direct Qwen Multi", a.internal_dir/"summary_qwen_multi_direct.csv"),
        ("WDD", a.internal_dir/"summary_weighted_degree_discount.csv"),
        ("RR-Greedy", a.internal_dir/"summary_rr_greedy.csv"),
        ("IMM-W1", a.imm_summary),
        ("WDD + VerifiedSearch-Proxy", a.wdd_proxy_summary),
        ("WDD + VerifiedSearch-LLM", a.wdd_llm_summary),
    ]
    frames=[read_method(path, method, node_map) for method,path in specs]
    long=pd.concat(frames, ignore_index=True, sort=False)
    long=long[(long["diffusion"].astype(str)=="weighted_cascade") &
              (pd.to_numeric(long["budget"], errors="coerce")==30)].copy()

    costs=candidate_costs(a.candidate_root)
    long=long.merge(costs, on=["graph_name","dataset_type","diffusion"], how="left")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(a.output_dir/"scalability_final_long.csv", index=False)

    wide=long.pivot_table(index="paper_method", columns="num_nodes",
                          values=["normalized_final_spread","final_eval_spread"], aggfunc="first")
    wide.columns=[f"{metric}_n{int(n)}" for metric,n in wide.columns]
    wide.reset_index().to_csv(a.output_dir/"scalability_final_wide.csv", index=False)

    prompt=pd.read_csv(a.prompt_summary).copy()
    prompt["status"]=prompt["all_fit_context"].astype(str).str.lower().map(
        lambda x: "Executable" if x=="true" else "Context overflow")
    cols=[c for c in ["graph_name","num_nodes","num_edges","mean_input_tokens","max_input_tokens",
                      "max_total_tokens_requested","model_context_limit","all_fit_context","status"]
          if c in prompt.columns]
    prompt[cols].sort_values("num_nodes").to_csv(a.output_dir/"scalability_context_table.csv", index=False)

    print(long[["paper_method","graph_name","num_nodes","normalized_final_spread",
                "final_eval_spread","reported_runtime_seconds",
                "candidate_generation_seconds_total"]]
          .sort_values(["num_nodes","paper_method"]).to_string(index=False))
    print(f"\nSaved to {a.output_dir}")

if __name__=="__main__":
    main()
