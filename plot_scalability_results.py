#!/usr/bin/env python3
"""Create paper-ready scalability figures from collected CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY = {
    "qwen_multi_direct": "Direct Qwen Multi",
    "proxy_fixed": "VerifiedSearch-Proxy",
    "llm_fixed_randomfill": "VerifiedSearch-LLM",
    "rr_greedy": "RR-Greedy",
    "imm_w1": "IMM-W1",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("VerifiedSearch/scalability/analysis"),
    )
    args = parser.parse_args()
    out = args.analysis_dir
    out.mkdir(parents=True, exist_ok=True)

    candidate = pd.read_csv(out / "scalability_candidate_cost.csv")
    candidate = candidate.sort_values("num_nodes")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        candidate["num_nodes"],
        candidate["mean_input_tokens"],
        marker="o",
    )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Mean input tokens")
    ax.set_title("Full-graph prompt size vs. graph size")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "prompt_tokens_vs_nodes.pdf", bbox_inches="tight")
    fig.savefig(
        out / "prompt_tokens_vs_nodes.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        candidate["num_nodes"],
        candidate["candidate_generation_seconds_total"],
        marker="o",
    )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Candidate generation time (s)")
    ax.set_title("Qwen candidate-generation scaling")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        out / "candidate_generation_time_vs_nodes.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out / "candidate_generation_time_vs_nodes.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    methods = pd.read_csv(out / "scalability_methods_long.csv")
    methods = methods[
        (methods["budget"] == 30)
        & (methods["diffusion"] == "weighted_cascade")
    ].copy()

    preferred = [
        "qwen_multi_direct",
        "proxy_fixed",
        "llm_fixed_randomfill",
        "rr_greedy",
        "imm_w1",
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for method in preferred:
        subset = methods[methods["method"] == method].sort_values(
            "num_nodes"
        )
        if subset.empty:
            continue
        ax.plot(
            subset["num_nodes"],
            subset["normalized_final_spread"],
            marker="o",
            label=DISPLAY.get(method, method),
        )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Normalized spread")
    ax.set_title("Influence quality under increasing graph size")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out / "normalized_spread_vs_nodes.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out / "normalized_spread_vs_nodes.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Created:")
    for path in sorted(out.glob("*.pdf")):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
