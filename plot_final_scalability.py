#!/usr/bin/env python3
"""Plot final IEEE BigData scalability results.

Reads:
  scalability_final_long.csv
  scalability_context_table.csv

Produces:
  normalized_spread_vs_nodes.pdf/png
  candidate_generation_time_vs_nodes.pdf/png
  prompt_tokens_vs_nodes.pdf/png
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PREFERRED = [
    "Direct Qwen Multi",
    "WDD",
    "WDD + VerifiedSearch-Proxy",
    "WDD + VerifiedSearch-LLM",
    "RR-Greedy",
    "IMM-W1",
]


def save(fig, outdir: Path, stem: str):
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=Path, required=True)
    a = p.parse_args()
    outdir = a.analysis_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(a.analysis_dir / "scalability_final_long.csv")
    context = pd.read_csv(a.analysis_dir / "scalability_context_table.csv")

    # 1) Quality vs size
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method in PREFERRED:
        sub = long[long["paper_method"] == method].sort_values("num_nodes")
        if sub.empty:
            continue
        ax.plot(
            sub["num_nodes"],
            sub["normalized_final_spread"],
            marker="o",
            label=method,
        )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Normalized influence spread")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, outdir, "normalized_spread_vs_nodes")

    # 2) Candidate-generation time vs size
    cost = (
        long[
            [
                "graph_name",
                "num_nodes",
                "candidate_generation_seconds_total",
            ]
        ]
        .drop_duplicates()
        .sort_values("num_nodes")
    )
    cost = cost[cost["candidate_generation_seconds_total"].notna()]
    if not cost.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(
            cost["num_nodes"],
            cost["candidate_generation_seconds_total"],
            marker="o",
        )
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("Qwen candidate-generation time (s)")
        ax.grid(True, alpha=0.25)
        save(fig, outdir, "candidate_generation_time_vs_nodes")

    # 3) Prompt tokens vs size, with context ceiling
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    context = context.sort_values("num_nodes")
    ax.plot(
        context["num_nodes"],
        context["mean_input_tokens"],
        marker="o",
        label="Mean input tokens",
    )
    if "model_context_limit" in context.columns:
        limit = float(context["model_context_limit"].dropna().iloc[0])
        ax.axhline(limit, linestyle="--", label=f"Context limit ({int(limit):,})")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Input tokens")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, outdir, "prompt_tokens_vs_nodes")

    print("Saved figures to:", outdir)


if __name__ == "__main__":
    main()
