#!/usr/bin/env python3
"""Measure exact Qwen candidate-generation prompt tokens without loading the model.

The script reuses VerifiedSearch/generate_candidates_qwen35.py's prompt
constructor and the same chat wrapper, then reports per-permutation and
per-graph token counts. Run this before expensive inference to identify the
full-graph serialization context limit.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import generate_candidates_qwen35 as gen


def resolve_path(raw: str | Path, manifest_path: Path) -> Path:
    try:
        return gen.resolve_manifest_path(raw, manifest_path)
    except Exception:
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path.resolve()
        candidates = [
            manifest_path.parent / path,
            manifest_path.parent.parent / path,
            Path.cwd() / path,
        ]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return candidate
        raise FileNotFoundError(raw)


def render_chat(tokenizer: Any, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise graph-optimization assistant. "
                "Follow the user's JSON output contract exactly."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    return (
        "System: You are a precise graph-optimization assistant.\n"
        f"User: {prompt}\nAssistant:"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "VerifiedSearch/processed_graphs_scalability/manifest.json"
        ),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "VerifiedSearch/scalability/prompt_tokens_by_permutation.csv"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "VerifiedSearch/scalability/prompt_tokens_by_graph.csv"
        ),
    )
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--list-multiplier", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--diffusion",
        choices=("uniform", "weighted_cascade"),
        default="weighted_cascade",
    )
    parser.add_argument("--uniform-p", type=float, default=0.1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    text_config = getattr(config, "text_config", None)
    model_limit = (
        getattr(config, "max_position_embeddings", None)
        or getattr(text_config, "max_position_embeddings", None)
    )
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 10**9:
        model_limit = (
            min(int(model_limit), tokenizer_limit)
            if model_limit is not None
            else tokenizer_limit
        )

    rows: list[dict[str, Any]] = []
    by_graph: dict[str, list[dict[str, Any]]] = {}

    for entry in manifest.get("graphs", []):
        graph_path = resolve_path(entry["processed_file"], manifest_path)
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        list_length = min(
            int(graph["num_nodes"]),
            args.list_multiplier * args.budget,
        )

        for raw_perm in entry.get("permutation_files", []):
            perm_path = resolve_path(raw_perm, manifest_path)
            text_path = perm_path.with_suffix(".txt")
            adjacency_text = text_path.read_text(encoding="utf-8")

            prompt = gen.build_candidate_prompt(
                adjacency_text=adjacency_text,
                graph_name=str(graph["name"]),
                num_nodes=int(graph["num_nodes"]),
                num_edges=int(graph["num_edges"]),
                list_length=list_length,
                max_budget=args.budget,
                diffusion=args.diffusion,
                uniform_p=args.uniform_p,
            )
            rendered = render_chat(tokenizer, prompt)
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            input_tokens = len(encoded["input_ids"])
            total_tokens = input_tokens + args.max_new_tokens
            fits_context = (
                ""
                if model_limit is None
                else total_tokens <= int(model_limit)
            )

            row = {
                "graph_name": graph["name"],
                "num_nodes": int(graph["num_nodes"]),
                "num_edges": int(graph["num_edges"]),
                "permutation": perm_path.stem,
                "adjacency_chars": len(adjacency_text),
                "input_tokens": input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "total_tokens_requested": total_tokens,
                "model_context_limit": model_limit,
                "fits_context": fits_context,
            }
            rows.append(row)
            by_graph.setdefault(str(graph["name"]), []).append(row)

    if not rows:
        raise SystemExit("No permutation records found in manifest")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for graph_name, graph_rows in sorted(
        by_graph.items(),
        key=lambda item: item[1][0]["num_nodes"],
    ):
        token_values = [int(row["input_tokens"]) for row in graph_rows]
        summary_rows.append(
            {
                "graph_name": graph_name,
                "num_nodes": graph_rows[0]["num_nodes"],
                "num_edges": graph_rows[0]["num_edges"],
                "permutations": len(graph_rows),
                "mean_input_tokens": round(statistics.fmean(token_values), 2),
                "min_input_tokens": min(token_values),
                "max_input_tokens": max(token_values),
                "max_total_tokens_requested": max(
                    int(row["total_tokens_requested"])
                    for row in graph_rows
                ),
                "model_context_limit": model_limit,
                "all_fit_context": (
                    ""
                    if model_limit is None
                    else all(bool(row["fits_context"]) for row in graph_rows)
                ),
            }
        )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Model context limit: {model_limit}")
    for row in summary_rows:
        print(
            f"{row['graph_name']:25s} "
            f"N={row['num_nodes']:>6} "
            f"mean={row['mean_input_tokens']:>10} "
            f"max={row['max_input_tokens']:>8} "
            f"fits={row['all_fit_context']}"
        )
    print(f"Per-permutation CSV: {args.output_csv}")
    print(f"Per-graph CSV:       {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
