#!/usr/bin/env python3
"""
Build final IEEE BigData result tables and figures for VerifiedSearch.

Run from:
    /mnt/guanghua/Project/IM

Examples:
    python VerifiedSearch/build_bigdata_results.py --target table1
    python VerifiedSearch/build_bigdata_results.py --target fig1a
    python VerifiedSearch/build_bigdata_results.py --target all

Default output:
    VerifiedSearch/paper_assets/

Final paper assets:
    Table I   Main benchmark
    Table II  Paired effects of verified refinement
    Table III Fixed-budget controller comparison
    Table IV  Budget-isolated WDD refinement ablation
    Table V   Scalability / context / LLM cost

    Fig. 1(a) Qwen-Multi -> VerifiedSearch, by diffusion
    Fig. 1(b) WDD -> VerifiedSearch, by diffusion
    Fig. 2(a) Quality vs graph size
    Fig. 2(b) Requested tokens vs graph size/context limit

Notes:
- Normalized spread is the primary cross-graph metric.
- Table IV uses the final budget-isolated ("clean") WDD experiments.
- Fig. 2(b) uses max_total_tokens_requested, not mean input tokens.
- Fig. 1 hides boxplot fliers because individual instances are already shown.
- "clean" is not used in paper-facing labels/captions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path("VerifiedSearch")
DEFAULT_OUT = Path("VerifiedSearch/paper_assets")

KEYS = [
    "graph_name",
    "dataset_type",
    "diffusion",
    "budget",
]

DIFF_UNIFORM = "uniform_ic_p0.1"
DIFF_WC = "weighted_cascade"


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def escape_tex(value) -> str:
    """Escape ordinary text for LaTeX table cells."""
    text = str(value)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def write_tex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    column_format: str | None = None,
    header_map: dict[str, str] | None = None,
    bold_cells: set[tuple[int, str]] | None = None,
    note: str | None = None,
    table_env: str = "table*",
) -> None:
    """
    Write a compact booktabs-style table.

    header_map values are treated as raw LaTeX.
    bold_cells contains (row_index, column_name) pairs.
    """
    header_map = header_map or {}
    bold_cells = bold_cells or set()

    cols = list(df.columns)
    if column_format is None:
        column_format = "l" + "r" * (len(cols) - 1)

    lines = [
        rf"\begin{{{table_env}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
    ]

    headers = []
    for col in cols:
        headers.append(header_map.get(col, escape_tex(col)))
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for ridx, row in df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                rendered = "--"
            else:
                rendered = escape_tex(value)

            if (ridx, col) in bold_cells and rendered != "--":
                rendered = rf"\textbf{{{rendered}}}"
            cells.append(rendered)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")

    if note:
        lines.append(
            rf"\multicolumn{{{len(cols)}}}{{l}}{{\footnotesize {note}}} \\"
        )

    lines.extend(
        [
            r"\end{tabular}",
            rf"\end{{{table_env}}}",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_optional(root: Path, patterns: list[str]) -> Path | None:
    """Find an optional result file using explicit relative paths or globs."""
    explicit = []
    globs = []

    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            globs.append(pattern)
        else:
            explicit.append(root / pattern)

    hit = first_existing(explicit)
    if hit is not None:
        return hit

    for pattern in globs:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]

    return None


def load_node_map(root: Path) -> pd.DataFrame:
    path = (
        root
        / "baseline_outputs_v2"
        / "internal"
        / "summary_qwen_multi_direct.csv"
    )
    df = pd.read_csv(path)

    required = set(KEYS + ["num_nodes"])
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{path} missing columns {sorted(missing)}"
        )

    node_map = df[KEYS + ["num_nodes"]].drop_duplicates()

    if node_map.duplicated(KEYS).any():
        raise RuntimeError("Node map is not one-to-one on instance keys.")

    return node_map


def load_with_normalized(
    path: Path,
    node_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Some normalized temporary files intentionally overwrite
    # final_eval_spread with normalized spread.
    if "normalized_final_spread" in df.columns:
        df["normalized_final_spread"] = pd.to_numeric(
            df["normalized_final_spread"],
            errors="coerce",
        )
        return df

    if "num_nodes" not in df.columns:
        if node_map is None:
            raise RuntimeError(
                f"{path}: num_nodes absent and no node map provided."
            )
        df = df.merge(
            node_map,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

    if "final_eval_spread" not in df.columns:
        raise RuntimeError(
            f"{path}: missing final_eval_spread."
        )

    # If this is one of our normalized temp files, final_eval_spread
    # may already be normalized. Detect it from raw_final_eval_spread.
    if "raw_final_eval_spread" in df.columns:
        df["normalized_final_spread"] = pd.to_numeric(
            df["final_eval_spread"],
            errors="coerce",
        )
    else:
        df["normalized_final_spread"] = (
            pd.to_numeric(df["final_eval_spread"], errors="raise")
            / pd.to_numeric(df["num_nodes"], errors="raise")
        )

    return df


def summarize_normalized(
    path: Path,
    node_map: pd.DataFrame | None = None,
) -> dict[str, float | int]:
    df = load_with_normalized(path, node_map)

    norm = pd.to_numeric(
        df["normalized_final_spread"],
        errors="coerce",
    )

    return {
        "coverage": int(norm.notna().sum()),
        "overall": float(norm.mean()),
        "uniform": float(
            pd.to_numeric(
                df.loc[
                    df["diffusion"].astype(str) == DIFF_UNIFORM,
                    "normalized_final_spread",
                ],
                errors="coerce",
            ).mean()
        ),
        "wc": float(
            pd.to_numeric(
                df.loc[
                    df["diffusion"].astype(str) == DIFF_WC,
                    "normalized_final_spread",
                ],
                errors="coerce",
            ).mean()
        ),
    }


def fmt4(x) -> str:
    return "--" if pd.isna(x) else f"{float(x):.4f}"


def print_saved(*paths: Path) -> None:
    for path in paths:
        print("Saved:", path)


# ---------------------------------------------------------------------
# Table I
# ---------------------------------------------------------------------

def table1(root: Path, outdir: Path) -> None:
    """
    Main benchmark:
      coverage + overall/uniform/WC normalized spread.

    Only full-coverage methods are eligible for bolded maxima.
    WC-only methods are shown with Overall/Uniform = --.
    """
    ensure_dir(outdir)

    internal = root / "baseline_outputs_v2" / "internal"
    node_map = load_node_map(root)

    specs: list[tuple[str, Path | None, bool]] = [
        (
            "Random",
            internal / "summary_random_seed_selection.csv",
            False,
        ),
        (
            "Direct Qwen Single",
            internal / "summary_qwen_single_direct.csv",
            False,
        ),
        (
            "Direct Qwen Multi",
            internal / "summary_qwen_multi_direct.csv",
            False,
        ),
        (
            "Out-Degree",
            internal / "summary_out_degree.csv",
            False,
        ),
        (
            "Qwen Multi + VerifiedSearch-LLM",
            root
            / "paper_analysis"
            / "normalized_tmp"
            / "qwen_verified.csv",
            False,
        ),
        (
            "OPIM-C",
            find_optional(
                root,
                [
                    "baseline_outputs_v2/external/summary_opim_c_official.csv",
                    "baseline_outputs_v2/**/summary*opim*.csv",
                ],
            ),
            False,
        ),
        (
            "WDD",
            internal / "summary_weighted_degree_discount.csv",
            False,
        ),
        (
            "WDD + VerifiedSearch-LLM",
            root
            / "wdd_candidate_control_analysis"
            / "normalized"
            / "summary_wdd_qwen_llm_clean_norm.csv",
            False,
        ),
        (
            "RR-Greedy",
            internal / "summary_rr_greedy.csv",
            False,
        ),
        (
            "IMM-W1",
            find_optional(
                root,
                [
                    "baseline_outputs_v2/imm/summary_imm_w1.csv",
                    "baseline_outputs_v2/**/summary*imm*w1*.csv",
                    "baseline_outputs_v2/**/summary*imm*.csv",
                ],
            ),
            False,
        ),
        (
            "ToupleGDD",
            find_optional(
                root,
                [
                    "baseline_outputs_v2/external/summary_touplegdd_official.csv",
                    "baseline_outputs_v2/**/summary*touple*.csv",
                ],
            ),
            True,
        ),
        (
            "S2V-DQN",
            find_optional(
                root,
                [
                    "baseline_outputs_v2/external/summary_s2vdqn_official.csv",
                    "baseline_outputs_v2/**/summary*s2v*.csv",
                ],
            ),
            True,
        ),
    ]

    rows = []
    numeric_rows = []

    for method, path, wc_only in specs:
        if path is None or not path.exists():
            print(f"[Table I] SKIP missing optional result: {method}")
            continue

        s = summarize_normalized(path, node_map)

        coverage_label = (
            f"{s['coverage']} (WC only)"
            if wc_only
            else str(s["coverage"])
        )

        rows.append(
            {
                "Method": method,
                "Coverage": coverage_label,
                "Overall": "--" if wc_only else fmt4(s["overall"]),
                "Uniform IC": "--" if wc_only else fmt4(s["uniform"]),
                "Weighted Cascade": fmt4(s["wc"]),
            }
        )

        numeric_rows.append(
            {
                "method": method,
                "coverage": int(s["coverage"]),
                "wc_only": wc_only,
                "overall": s["overall"],
                "uniform": s["uniform"],
                "wc": s["wc"],
            }
        )

    df = pd.DataFrame(rows)

    # Bold best FULL-COVERAGE comparable result per metric.
    bold_cells: set[tuple[int, str]] = set()
    num = pd.DataFrame(numeric_rows)

    eligible = num[
        (~num["wc_only"])
        & (num["coverage"] == 162)
    ]

    metric_map = {
        "Overall": "overall",
        "Uniform IC": "uniform",
        "Weighted Cascade": "wc",
    }

    for display_col, numeric_col in metric_map.items():
        if eligible.empty:
            continue
        best = eligible[numeric_col].max()
        winners = set(
            eligible.loc[
                np.isclose(eligible[numeric_col], best),
                "method",
            ]
        )
        for ridx, row in df.iterrows():
            if row["Method"] in winners:
                bold_cells.add((ridx, display_col))

    csv_path = outdir / "table1_main_benchmark.csv"
    tex_path = outdir / "table1_main_benchmark.tex"

    df.to_csv(csv_path, index=False)

    write_tex_table(
        df,
        tex_path,
        caption=(
            "Mean normalized influence spread on the main benchmark. "
            "Bold values indicate the best full-coverage method in each column."
        ),
        label="tab:main_benchmark",
        column_format="lrrrr",
        bold_cells=bold_cells,
    )

    print(df.to_string(index=False))
    print_saved(csv_path, tex_path)


# ---------------------------------------------------------------------
# Table II
# ---------------------------------------------------------------------

def paired_stat_rows(
    path: Path,
    initialization: str,
) -> list[dict[str, str]]:
    df = pd.read_csv(path)

    wanted = [
        ("overall", "Overall"),
        ("diffusion:uniform_ic_p0.1", "Uniform IC"),
        ("diffusion:weighted_cascade", "Weighted Cascade"),
    ]

    rows = []

    for key, setting in wanted:
        hit = df[df["group"].astype(str) == key]
        if hit.empty:
            raise RuntimeError(
                f"{path}: missing statistics row {key}"
            )

        r = hit.iloc[0]

        p = float(r["p_signflip_holm"])
        ptxt = "<.001" if p < 0.001 else f"{p:.3f}"

        rows.append(
            {
                "Initialization": initialization,
                "Setting": setting,
                "Before": f"{float(r['direct_mean']):.4f}",
                "After": f"{float(r['verified_mean']):.4f}",
                "Delta": f"{float(r['mean_difference']):+.4f}",
                "Rel. Delta": (
                    f"{float(r['relative_improvement_percent']):+.2f}%"
                ),
                "95% CI": (
                    f"[{float(r['bootstrap_ci95_low']):.4f}, "
                    f"{float(r['bootstrap_ci95_high']):.4f}]"
                ),
                "W/T/L": (
                    f"{int(r['wins_verified'])}/"
                    f"{int(r['ties'])}/"
                    f"{int(r['losses_verified'])}"
                ),
                "Holm p": ptxt,
            }
        )

    return rows


def table2(root: Path, outdir: Path) -> None:
    ensure_dir(outdir)

    qwen_stats = (
        root
        / "paper_analysis"
        / "verified_vs_direct_normalized"
        / "verified_vs_direct_stats.csv"
    )

    wdd_stats = (
        root
        / "wdd_candidate_control_analysis"
        / "normalized"
        / "stats_clean_llm_vs_wdd"
        / "verified_vs_direct_stats.csv"
    )

    rows = (
        paired_stat_rows(qwen_stats, "Qwen-Multi")
        + paired_stat_rows(wdd_stats, "WDD")
    )

    df = pd.DataFrame(rows)

    csv_path = outdir / "table2_paired_effects.csv"
    tex_path = outdir / "table2_paired_effects.tex"

    df.to_csv(csv_path, index=False)

    write_tex_table(
        df,
        tex_path,
        caption=(
            "Paired effect of verified refinement on normalized influence "
            "spread. Confidence intervals are 95\\% bootstrap intervals for "
            "the mean paired difference; $p_{\\mathrm{Holm}}$ denotes "
            "Holm-corrected sign-flip $p$-values."
        ),
        label="tab:paired_effects",
        column_format="llrrrrrrr",
        header_map={
            "Delta": r"$\Delta$",
            "Rel. Delta": r"Rel. $\Delta$",
            "95% CI": r"95\% CI",
            "Holm p": r"$p_{\mathrm{Holm}}$",
        },
    )

    print(df.to_string(index=False))
    print_saved(csv_path, tex_path)


# ---------------------------------------------------------------------
# Table III
# ---------------------------------------------------------------------

def table3(root: Path, outdir: Path) -> None:
    ensure_dir(outdir)

    base = root / "verified_search_fixed_budget_outputs"
    node_map = load_node_map(root)

    specs = [
        ("Random", base / "summary_random_fixed.csv"),
        ("LLM", base / "summary_llm_fixed_randomfill.csv"),
        ("Proxy", base / "summary_proxy_fixed.csv"),
        ("Exhaustive-Swap", base / "summary_exhaustive_fixed.csv"),
    ]

    values = {}

    for method, path in specs:
        df = pd.read_csv(path)

        if "num_nodes" not in df.columns:
            df = df.merge(
                node_map,
                on=KEYS,
                how="left",
                validate="one_to_one",
            )

        df["normalized"] = (
            pd.to_numeric(df["final_eval_spread"], errors="raise")
            / pd.to_numeric(df["num_nodes"], errors="raise")
        )

        values[method] = {
            "norm": float(df["normalized"].mean()),
            "raw": float(
                pd.to_numeric(
                    df["final_eval_spread"],
                    errors="coerce",
                ).mean()
            ),
            "gain": float(
                pd.to_numeric(
                    df["gain_over_initialization"],
                    errors="coerce",
                ).mean()
            ),
            "verified": float(
                pd.to_numeric(
                    df["verified_swaps"],
                    errors="coerce",
                ).mean()
            ),
        }

    exhaustive = values["Exhaustive-Swap"]["verified"]

    rows = []

    for method, _ in specs:
        d = values[method]

        rows.append(
            {
                "Controller": method,
                "Norm. Spread": f"{d['norm']:.4f}",
                "Raw Spread": f"{d['raw']:.2f}",
                "Raw Gain": f"{d['gain']:.2f}",
                "Verified Swaps": f"{d['verified']:.1f}",
                "Relative Verification": (
                    f"{100.0 * d['verified'] / exhaustive:.1f}%"
                ),
            }
        )

    df = pd.DataFrame(rows)

    csv_path = outdir / "table3_controller_analysis.csv"
    tex_path = outdir / "table3_controller_analysis.tex"

    df.to_csv(csv_path, index=False)

    write_tex_table(
        df,
        tex_path,
        caption=(
            "Controller comparison under a fixed verification budget. "
            "Relative verification is normalized to the 90 swap evaluations "
            "used by Exhaustive-Swap."
        ),
        label="tab:controller",
        column_format="lrrrrr",
    )

    print(df.to_string(index=False))
    print_saved(csv_path, tex_path)


# ---------------------------------------------------------------------
# Table IV
# ---------------------------------------------------------------------

def mean_by_diffusion_from_normalized_file(
    path: Path,
) -> tuple[float, float, float]:
    df = pd.read_csv(path)

    # These files intentionally store normalized spread in final_eval_spread.
    if "final_eval_spread" not in df.columns:
        raise RuntimeError(
            f"{path}: missing final_eval_spread"
        )

    col = pd.to_numeric(
        df["final_eval_spread"],
        errors="raise",
    )

    overall = float(col.mean())

    uniform = float(
        pd.to_numeric(
            df.loc[
                df["diffusion"].astype(str) == DIFF_UNIFORM,
                "final_eval_spread",
            ],
            errors="coerce",
        ).mean()
    )

    wc = float(
        pd.to_numeric(
            df.loc[
                df["diffusion"].astype(str) == DIFF_WC,
                "final_eval_spread",
            ],
            errors="coerce",
        ).mean()
    )

    return overall, uniform, wc


def table4(root: Path, outdir: Path) -> None:
    ensure_dir(outdir)

    normdir = (
        root
        / "wdd_candidate_control_analysis"
        / "normalized"
    )

    specs = [
        (
            "WDD",
            normdir / "wdd_norm.csv",
        ),
        (
            "WDD + Qwen + Proxy",
            normdir / "summary_wdd_qwen_proxy_clean_norm.csv",
        ),
        (
            "WDD + Qwen + LLM",
            normdir / "summary_wdd_qwen_llm_clean_norm.csv",
        ),
        (
            "WDD + Out-Degree + Proxy",
            normdir / "summary_wdd_out_degree_proxy_clean_norm.csv",
        ),
        (
            "WDD + Hybrid + Proxy",
            normdir / "summary_wdd_hybrid_proxy_clean_norm.csv",
        ),
    ]

    values = {
        method: mean_by_diffusion_from_normalized_file(path)
        for method, path in specs
    }

    wdd_overall = values["WDD"][0]

    rows = []

    for method, _ in specs:
        overall, uniform, wc = values[method]

        if method == "WDD":
            rel = "--"
        else:
            rel = (
                f"{100.0 * (overall - wdd_overall) / wdd_overall:+.2f}%"
            )

        rows.append(
            {
                "Method": method,
                "Overall Norm.": f"{overall:.4f}",
                "Uniform IC": f"{uniform:.4f}",
                "Weighted Cascade": f"{wc:.4f}",
                "Overall Delta vs WDD": rel,
            }
        )

    df = pd.DataFrame(rows)

    csv_path = outdir / "table4_wdd_ablation.csv"
    tex_path = outdir / "table4_wdd_ablation.tex"

    df.to_csv(csv_path, index=False)

    write_tex_table(
        df,
        tex_path,
        caption=(
            "Budget-isolated candidate-source and controller ablation with "
            "WDD initialization. No candidate-source comparison is "
            "significant after Holm correction."
        ),
        label="tab:wdd_ablation",
        column_format="lrrrr",
        header_map={
            "Overall Delta vs WDD": r"Overall $\Delta$ vs. WDD",
        },
    )

    print(df.to_string(index=False))
    print_saved(csv_path, tex_path)


# ---------------------------------------------------------------------
# Table V
# ---------------------------------------------------------------------

def table5(root: Path, outdir: Path) -> None:
    ensure_dir(outdir)

    final = root / "scalability" / "final_analysis"

    long_path = final / "scalability_final_long.csv"
    context_path = final / "scalability_context_table.csv"

    long = pd.read_csv(long_path)
    context = pd.read_csv(context_path)

    cost_cols = [
        "graph_name",
        "num_nodes",
        "candidate_generation_seconds_total",
        "candidate_llm_calls",
        "candidate_input_tokens_mean_per_permutation",
    ]

    missing_cost_cols = set(cost_cols) - set(long.columns)
    if missing_cost_cols:
        raise RuntimeError(
            f"{long_path} missing {sorted(missing_cost_cols)}"
        )

    # Cost is graph-level and duplicated across method rows.
    costs = (
        long[cost_cols]
        .drop_duplicates()
        .sort_values("num_nodes")
    )

    if costs.duplicated(["graph_name", "num_nodes"]).any():
        raise RuntimeError(
            "Scalability candidate cost is not unique per graph."
        )

    merged = context.merge(
        costs,
        on=["graph_name", "num_nodes"],
        how="left",
        validate="one_to_one",
    )

    rows = []

    for _, r in merged.sort_values("num_nodes").iterrows():
        calls = r.get("candidate_llm_calls")
        total_sec = r.get("candidate_generation_seconds_total")

        if (
            pd.notna(total_sec)
            and pd.notna(calls)
            and float(calls) > 0
        ):
            avg_sec = float(total_sec) / float(calls)
        else:
            avg_sec = np.nan

        calls_text = "--"
        if pd.notna(calls):
            calls_int = int(calls)
            calls_text = (
                f"{calls_int}*"
                if int(r["num_nodes"]) == 10000 and calls_int == 6
                else str(calls_int)
            )

        rows.append(
            {
                "Nodes": f"{int(r['num_nodes']):,}",
                "Edges": f"{int(r['num_edges']):,}",
                "Mean Prompt Tokens": (
                    f"{float(r['mean_input_tokens']) / 1000:.1f}K"
                ),
                "LLM Calls": calls_text,
                "Candidate Gen. Time": (
                    "--"
                    if pd.isna(total_sec)
                    else f"{float(total_sec):.1f} s"
                ),
                "Avg. Time/Call": (
                    "--"
                    if pd.isna(avg_sec)
                    else f"{float(avg_sec):.1f} s"
                ),
                "Max Requested Tokens": (
                    f"{float(r['max_total_tokens_requested']) / 1000:.1f}K"
                ),
                "Status": str(r["status"]),
            }
        )

    df = pd.DataFrame(rows)

    csv_path = outdir / "table5_scalability_cost.csv"
    tex_path = outdir / "table5_scalability_cost.tex"

    df.to_csv(csv_path, index=False)

    write_tex_table(
        df,
        tex_path,
        caption=(
            "Full-graph LLM serialization and candidate-generation cost."
        ),
        label="tab:scalability_cost",
        column_format="lrrrrrrl",
        note=(
            r"* The 10K run contains one additional LLM call due to a "
            r"generation retry."
        ),
    )

    print(df.to_string(index=False))
    print_saved(csv_path, tex_path)


# ---------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------

def infer_paired_columns(
    df: pd.DataFrame,
) -> tuple[str, str]:
    candidates = [
        ("direct_spread", "verified_spread"),
        ("direct_final_eval_spread", "verified_final_eval_spread"),
        ("direct", "verified"),
    ]

    for left, right in candidates:
        if left in df.columns and right in df.columns:
            return left, right

    raise RuntimeError(
        "Cannot infer paired spread columns. "
        f"Available columns: {list(df.columns)}"
    )


def plot_delta_panel(
    paired_path: Path,
    out_pdf: Path,
    out_png: Path,
) -> None:
    df = pd.read_csv(paired_path)

    direct_col, verified_col = infer_paired_columns(df)

    df["delta"] = (
        pd.to_numeric(df[verified_col], errors="raise")
        - pd.to_numeric(df[direct_col], errors="raise")
    )

    groups = [
        (DIFF_UNIFORM, "Uniform IC"),
        (DIFF_WC, "Weighted Cascade"),
    ]

    data = []
    labels = []

    for key, label in groups:
        values = (
            df.loc[
                df["diffusion"].astype(str) == key,
                "delta",
            ]
            .astype(float)
            .to_numpy()
        )
        data.append(values)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(5.8, 4.1))

    ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        showfliers=False,
    )

    rng = np.random.default_rng(2026)

    for idx, values in enumerate(data, start=1):
        x = rng.normal(
            idx,
            0.035,
            size=len(values),
        )
        ax.scatter(
            x,
            values,
            s=10,
            alpha=0.35,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_ylabel(
        "Change in normalized influence spread"
    )

    # No internal title. Paper caption/subcaption will identify the panel.
    ax.grid(
        True,
        axis="y",
        alpha=0.18,
    )

    fig.tight_layout()

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 1(a)
# ---------------------------------------------------------------------

def fig1a(root: Path, outdir: Path) -> None:
    ensure_dir(outdir)

    paired = (
        root
        / "paper_analysis"
        / "verified_vs_direct_normalized"
        / "verified_vs_direct_paired_instances.csv"
    )

    pdf = outdir / "figure1a_qwen_verified_gain.pdf"
    png = outdir / "figure1a_qwen_verified_gain.png"

    plot_delta_panel(
        paired,
        pdf,
        png,
    )

    print_saved(pdf, png)


# ---------------------------------------------------------------------
# Figure 1(b)
# ---------------------------------------------------------------------

def fig1b(root: Path, outdir: Path) -> None:
    ensure_dir(outdir)

    paired = (
        root
        / "wdd_candidate_control_analysis"
        / "normalized"
        / "stats_clean_llm_vs_wdd"
        / "verified_vs_direct_paired_instances.csv"
    )

    pdf = outdir / "figure1b_wdd_verified_gain.pdf"
    png = outdir / "figure1b_wdd_verified_gain.png"

    plot_delta_panel(
        paired,
        pdf,
        png,
    )

    print_saved(pdf, png)


# ---------------------------------------------------------------------
# Figure 2(a)
# ---------------------------------------------------------------------

def fig2a(root: Path, outdir: Path) -> None:
    ensure_dir(outdir)

    path = (
        root
        / "scalability"
        / "final_analysis"
        / "scalability_final_long.csv"
    )

    df = pd.read_csv(path)

    methods = [
        ("Direct Qwen Multi", "Qwen-Multi"),
        ("WDD", "WDD"),
        (
            "WDD + VerifiedSearch-Proxy",
            "WDD + VerifiedSearch",
        ),
        ("RR-Greedy", "RR-Greedy"),
        ("IMM-W1", "IMM-W1"),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    observed_sizes = set()

    for method, label in methods:
        sub = df[
            df["paper_method"].astype(str) == method
        ].sort_values("num_nodes")

        if sub.empty:
            print(f"[Fig. 2a] SKIP missing method: {method}")
            continue

        observed_sizes.update(
            pd.to_numeric(
                sub["num_nodes"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

        ax.plot(
            sub["num_nodes"],
            sub["normalized_final_spread"],
            marker="o",
            linewidth=2,
            label=label,
        )

    sizes = sorted(observed_sizes)

    ax.set_xticks(sizes)

    ax.set_xticklabels(
        [
            f"{n // 1000}K"
            if n >= 1000 and n % 1000 == 0
            else str(n)
            for n in sizes
        ]
    )

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Normalized influence spread")

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        frameon=True,
    )

    fig.tight_layout()

    pdf = outdir / "figure2a_scalability_quality.pdf"
    png = outdir / "figure2a_scalability_quality.png"

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print_saved(pdf, png)


# ---------------------------------------------------------------------
# Figure 2(b)
# ---------------------------------------------------------------------

def fig2b(root: Path, outdir: Path) -> None:
    """
    Plot the quantity that actually determines context feasibility:
        max_total_tokens_requested
    against the model context limit.
    """
    ensure_dir(outdir)

    path = (
        root
        / "scalability"
        / "final_analysis"
        / "scalability_context_table.csv"
    )

    df = pd.read_csv(path).sort_values("num_nodes")

    required = {
        "num_nodes",
        "max_total_tokens_requested",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path} missing {sorted(missing)}"
        )

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(
        df["num_nodes"],
        df["max_total_tokens_requested"],
        marker="o",
        linewidth=2,
        label="Requested tokens",
    )

    if (
        "model_context_limit" in df.columns
        and df["model_context_limit"].notna().any()
    ):
        limit = float(
            df["model_context_limit"]
            .dropna()
            .iloc[0]
        )

        ax.axhline(
            limit,
            linestyle="--",
            linewidth=1.8,
            label=f"Context limit ({int(limit):,})",
        )

    if "status" in df.columns:
        overflow = df[
            df["status"]
            .astype(str)
            .str.contains(
                "overflow",
                case=False,
                na=False,
            )
        ]

        for _, row in overflow.iterrows():
            ax.annotate(
                "Context overflow",
                xy=(
                    float(row["num_nodes"]),
                    float(
                        row["max_total_tokens_requested"]
                    ),
                ),
                xytext=(-12, -20),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=9,
            )

    nodes = (
        pd.to_numeric(
            df["num_nodes"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    ax.set_xticks(nodes)

    ax.set_xticklabels(
        [
            f"{n // 1000}K"
            if n >= 1000 and n % 1000 == 0
            else str(n)
            for n in nodes
        ]
    )

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Requested tokens")

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        frameon=True,
    )

    # Give the right edge a little room so the overflow annotation stays inside.
    if len(nodes) >= 2:
        span = max(nodes) - min(nodes)
        ax.set_xlim(
            min(nodes) - 0.04 * span,
            max(nodes) + 0.10 * span,
        )

    fig.tight_layout()

    pdf = outdir / "figure2b_context_scaling.pdf"
    png = outdir / "figure2b_context_scaling.png"

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print_saved(pdf, png)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

TARGETS = {
    "table1": table1,
    "table2": table2,
    "table3": table3,
    "table4": table4,
    "table5": table5,
    "fig1a": fig1a,
    "fig1b": fig1b,
    "fig2a": fig2a,
    "fig2b": fig2b,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--target",
        required=True,
        choices=list(TARGETS) + ["all"],
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    if args.target == "all":
        for name, fn in TARGETS.items():
            print()
            print("=" * 72)
            print(name)
            print("=" * 72)
            fn(args.root, args.output_dir)
    else:
        TARGETS[args.target](
            args.root,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
