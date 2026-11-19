#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

DISPLAY = {
    "imm_w1": "IMM-W1",
    "rr_greedy": "RR-Greedy",
    "weighted_degree_discount": "Weighted Degree Discount",
    "degree_discount_ic_directed": "Degree Discount IC",
    "opim_c_official": "OPIM-C",
    "weighted_out_degree": "Weighted Out-Degree",
    "reverse_pagerank": "Reverse PageRank",
    "out_degree": "Out-Degree",
    "exhaustive_fixed": "Exhaustive-Swap",
    "proxy_fixed": "VerifiedSearch-Proxy",
    "llm_fixed_randomfill": "VerifiedSearch-LLM",
    "random_fixed": "VerifiedSearch-Random",
    "qwen_multi_direct": "Direct Qwen Multi",
    "qwen_single_direct": "Direct Qwen Single",
    "Qwen_Qwen3.5-9B_mv10": "Direct Qwen MV@10",
    "random_seed_selection": "Random",
    "touplegdd_official": "ToupleGDD",
    "s2vdqn_official": "S2V-DQN",
}

def col(df, names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"Missing one of {names}; columns={list(df.columns)}")
    return None

def read_overall(path, prefix):
    if not path.exists():
        return pd.DataFrame(columns=["method", f"{prefix}_spread", f"{prefix}_norm"])
    df = pd.read_csv(path)
    m = col(df, ["method", "controller"])
    s = col(df, ["mean_final_spread", "final_eval_spread", "final_spread"])
    n = col(df, ["mean_normalized_spread", "normalized_final_spread", "normalized_spread"], False)
    out = pd.DataFrame({
        "method": df[m].astype(str),
        f"{prefix}_spread": pd.to_numeric(df[s], errors="coerce"),
    })
    out[f"{prefix}_norm"] = pd.to_numeric(df[n], errors="coerce") if n else pd.NA
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-instance", type=Path, default=Path("VerifiedSearch/baseline_analysis_full162/unified_instance_results.csv"))
    ap.add_argument("--uniform-overall", type=Path, default=Path("VerifiedSearch/baseline_analysis_uniform81/unified_overall.csv"))
    ap.add_argument("--weighted-overall", type=Path, default=Path("VerifiedSearch/baseline_analysis_weighted81/unified_overall.csv"))
    ap.add_argument("--scalability-long", type=Path, default=Path("VerifiedSearch/scalability/analysis/scalability_methods_long.csv"))
    ap.add_argument("--prompt-summary", type=Path, default=Path("VerifiedSearch/scalability/prompt_tokens_by_graph.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("VerifiedSearch/integrated_analysis"))
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    main = pd.read_csv(a.main_instance)
    mm = col(main, ["method", "controller"])
    ms = col(main, ["final_eval_spread", "final_spread"])
    mn = col(main, ["normalized_final_spread", "normalized_spread"], False)
    agg = {ms: "mean"}
    if mn:
        agg[mn] = "mean"
    overall = main.groupby(mm, as_index=False).agg(agg).rename(columns={mm:"method", ms:"main_overall_spread"})
    if mn:
        overall = overall.rename(columns={mn:"main_overall_norm"})
    else:
        overall["main_overall_norm"] = pd.NA

    integrated = overall.merge(read_overall(a.uniform_overall, "uniform"), on="method", how="outer")
    integrated = integrated.merge(read_overall(a.weighted_overall, "weighted"), on="method", how="outer")

    scale = pd.read_csv(a.scalability_long)
    sm = col(scale, ["method", "controller"])
    sb = col(scale, ["budget"])
    sd = col(scale, ["diffusion"])
    sn = col(scale, ["num_nodes"])
    ss = col(scale, ["final_eval_spread", "final_spread"])
    sx = col(scale, ["normalized_final_spread", "normalized_spread"], False)
    sr = col(scale, ["runtime_seconds", "selection_seconds", "elapsed_seconds"], False)

    scale = scale[(pd.to_numeric(scale[sb], errors="coerce") == 30) & (scale[sd].astype(str) == "weighted_cascade")].copy()
    scale["method"] = scale[sm].astype(str)
    scale["num_nodes_int"] = pd.to_numeric(scale[sn], errors="coerce").astype("Int64")

    for n in sorted(scale["num_nodes_int"].dropna().unique()):
        sub = scale[scale["num_nodes_int"] == n].copy()
        label = f"scale_{int(n)//1000}k" if int(n) % 1000 == 0 else f"scale_{int(n)}"
        part = pd.DataFrame({
            "method": sub["method"],
            f"{label}_spread": pd.to_numeric(sub[ss], errors="coerce"),
            f"{label}_norm": pd.to_numeric(sub[sx], errors="coerce") if sx else pd.NA,
            f"{label}_runtime_s": pd.to_numeric(sub[sr], errors="coerce") if sr else pd.NA,
        })
        integrated = integrated.merge(part, on="method", how="outer")

    integrated["display_name"] = integrated["method"].map(lambda x: DISPLAY.get(str(x), str(x)))
    integrated = integrated.sort_values("main_overall_norm", ascending=False, na_position="last")
    integrated.to_csv(a.output_dir / "integrated_method_table.csv", index=False)

    prompt = pd.read_csv(a.prompt_summary)
    keep = [c for c in ["graph_name","num_nodes","num_edges","mean_input_tokens","max_input_tokens","max_total_tokens_requested","model_context_limit","all_fit_context"] if c in prompt.columns]
    feas = prompt[keep].copy()
    feas["status"] = feas["all_fit_context"].astype(str).str.lower().map(lambda x: "Executable" if x == "true" else "Context overflow")
    feas.to_csv(a.output_dir / "scalability_feasibility_table.csv", index=False)

    main_long = main.copy()
    main_long["experiment_group"] = "main_benchmark"
    scale_long = pd.read_csv(a.scalability_long)
    scale_long["experiment_group"] = "scalability_stress"
    pd.concat([main_long, scale_long], ignore_index=True, sort=False).to_csv(a.output_dir / "combined_instance_results.csv", index=False)

    cols = [c for c in ["display_name","main_overall_spread","main_overall_norm","uniform_spread","weighted_spread","scale_2k_norm","scale_5k_norm","scale_10k_norm"] if c in integrated.columns]
    integrated[cols].to_latex(a.output_dir / "integrated_method_table.tex", index=False, float_format="%.4f", na_rep="--", escape=False)

    print("Wrote:")
    for p in sorted(a.output_dir.iterdir()):
        print(" ", p)

if __name__ == "__main__":
    main()
