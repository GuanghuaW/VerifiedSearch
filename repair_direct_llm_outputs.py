#!/usr/bin/env python3
"""Repair cached Direct-LLM runs by reparsing saved raw model outputs.

This script does not call the model. It only revisits the raw outputs already
stored in direct_llm_runs.json files, using a more tolerant parser that can
recover:
- fenced JSON;
- Python-style dicts/lists;
- "Node 17" and "#17" entries;
- malformed or truncated seed_nodes/ranked_candidates lists;
- bare comma-separated node lists.

It creates a .before_reparse backup before changing each instance file.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path
from typing import Any


EXPECTED_KEYS = (
    "seed_nodes",
    "seeds",
    "ranked_candidates",
    "nodes",
    "answer",
)


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.I | re.S)
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|python)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def normalize_node(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = re.fullmatch(
        r"\s*(?:node\s*)?[#:]?\s*([+-]?\d+)\s*",
        str(value),
        flags=re.I,
    )
    return int(match.group(1)) if match else None


def extract_payload(text: str) -> tuple[Any, str]:
    cleaned = strip_thinking(text)

    # Complete JSON/Python literal.
    for loader_name, loader in (("json", json.loads), ("literal", ast.literal_eval)):
        try:
            return loader(cleaned), loader_name
        except Exception:
            pass

    # First complete object/list embedded in surrounding prose.
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        fragment = cleaned[index:]
        try:
            value, _ = decoder.raw_decode(fragment)
            return value, "embedded_json"
        except Exception:
            try:
                return ast.literal_eval(fragment), "embedded_literal"
            except Exception:
                pass

    # Recover a malformed/truncated keyed list. We deliberately parse only the
    # text after a known output key, avoiding unrelated numbers in prose.
    key_group = "|".join(map(re.escape, EXPECTED_KEYS))
    marker = re.search(
        rf"[\"']?(?:{key_group})[\"']?\s*[:=]\s*\[",
        cleaned,
        flags=re.I,
    )
    if marker:
        tail = cleaned[marker.end():]
        tail = tail.split("]", 1)[0]
        tokens = re.findall(
            r"(?:node\s*)?[#:]?\s*([+-]?\d+)",
            tail,
            flags=re.I,
        )
        if tokens:
            return [int(token) for token in tokens], "truncated_keyed_list"

    # Recover a complete bare bracketed list.
    bare = re.search(r"\[([^\]]+)\]", cleaned, flags=re.S)
    if bare:
        tokens = re.findall(
            r"(?:node\s*)?[#:]?\s*([+-]?\d+)",
            bare.group(1),
            flags=re.I,
        )
        if tokens:
            return [int(token) for token in tokens], "bare_list"

    # Recover a response that is only a comma/space separated list of integers.
    # Require at least two numbers and reject alphabetic prose other than Node.
    simplified = re.sub(r"\bnode\b", "", cleaned, flags=re.I)
    numbers = re.findall(r"(?<![\w.])[+-]?\d+(?![\w.])", simplified)
    residual = re.sub(r"(?<![\w.])[+-]?\d+(?![\w.])", "", simplified)
    residual = re.sub(r"[\s,;:#.\-]+", "", residual)
    if len(numbers) >= 2 and not residual:
        return [int(value) for value in numbers], "bare_sequence"

    raise ValueError("No recoverable node list found.")


def parse_nodes(
    raw_output: str,
    *,
    num_nodes: int,
    desired_length: int,
) -> tuple[list[int], dict[str, Any]]:
    value, parser = extract_payload(raw_output)

    if isinstance(value, dict):
        items = None
        for key in EXPECTED_KEYS:
            if isinstance(value.get(key), (list, tuple)):
                items = value[key]
                break
        if items is None:
            raise ValueError("Parsed object has no recognized node-list key.")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise ValueError("Parsed value is not a list or recognized object.")

    result: list[int] = []
    invalid: list[Any] = []
    out_of_range: list[int] = []
    duplicates: list[int] = []
    seen: set[int] = set()

    for item in items:
        if isinstance(item, dict):
            item = item.get("node", item.get("id"))
        node = normalize_node(item)
        if node is None:
            invalid.append(item)
            continue
        if not 1 <= node <= num_nodes:
            out_of_range.append(node)
            continue
        if node in seen:
            duplicates.append(node)
            continue
        seen.add(node)
        result.append(node)
        if len(result) == desired_length:
            break

    return result, {
        "parser": parser,
        "parsed_count": len(result),
        "desired_length": desired_length,
        "invalid_items": invalid,
        "out_of_range_items": out_of_range,
        "duplicate_items": duplicates,
    }


def infer_num_nodes(payload: dict[str, Any]) -> int:
    # Most processed graph JSON files contain num_nodes or nodes.
    graph_file = Path(payload["graph_file"])
    graph_data = json.loads(graph_file.read_text(encoding="utf-8"))
    if isinstance(graph_data.get("num_nodes"), int):
        return int(graph_data["num_nodes"])
    nodes = graph_data.get("nodes")
    if isinstance(nodes, list):
        if nodes and isinstance(nodes[0], dict):
            ids = [
                int(node.get("id", node.get("node")))
                for node in nodes
                if node.get("id", node.get("node")) is not None
            ]
            if ids:
                return max(ids)
        if nodes:
            return max(map(int, nodes))
    adjacency = graph_data.get("adjacency")
    if isinstance(adjacency, dict):
        return max(map(int, adjacency))
    raise ValueError(f"Cannot infer num_nodes from {graph_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing per-instance direct_llm_runs.json files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show recoveries without modifying files.",
    )
    args = parser.parse_args()

    files = sorted(args.root.rglob("direct_llm_runs.json"))
    if not files:
        raise SystemExit(f"No direct_llm_runs.json files found under {args.root}")

    recovered_calls = 0
    changed_files = 0
    remaining: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        num_nodes = infer_num_nodes(payload)
        file_changed = False

        for budget_text, calls in payload.get("runs_by_budget", {}).items():
            budget = int(budget_text)
            for call in calls:
                if call.get("valid") and len(call.get("seed_nodes", [])) >= budget:
                    continue

                best_nodes = list(map(int, call.get("seed_nodes", [])))
                best_diag: dict[str, Any] = {
                    "parser": "stored",
                    "parsed_count": len(best_nodes),
                    "desired_length": budget,
                }
                best_attempt_index = None

                for attempt_index, attempt in enumerate(call.get("attempts", [])):
                    raw = str(attempt.get("raw_output", ""))
                    try:
                        nodes, diag = parse_nodes(
                            raw,
                            num_nodes=num_nodes,
                            desired_length=budget,
                        )
                    except Exception:
                        continue
                    if len(nodes) > len(best_nodes):
                        best_nodes = nodes
                        best_diag = diag
                        best_attempt_index = attempt_index

                if len(best_nodes) >= budget:
                    call["seed_nodes"] = best_nodes[:budget]
                    call["valid"] = True
                    call["reparsed"] = True
                    call["reparse_diagnostics"] = best_diag
                    call["reparse_attempt_index"] = best_attempt_index
                    recovered_calls += 1
                    file_changed = True
                    details.append(
                        {
                            "graph_name": payload.get("graph_name"),
                            "diffusion": payload.get("diffusion"),
                            "budget": budget,
                            "run_index": call.get("run_index"),
                            "parser": best_diag.get("parser"),
                        }
                    )
                else:
                    remaining.append(
                        {
                            "graph_name": payload.get("graph_name"),
                            "diffusion": payload.get("diffusion"),
                            "budget": budget,
                            "run_index": call.get("run_index"),
                            "best_parsed_count": len(best_nodes),
                            "instance_file": str(path),
                        }
                    )

        if file_changed:
            changed_files += 1
            if not args.dry_run:
                backup = path.with_name(path.name + ".before_reparse")
                if not backup.exists():
                    shutil.copy2(path, backup)
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    report = {
        "files_scanned": len(files),
        "changed_files": changed_files,
        "recovered_calls": recovered_calls,
        "remaining_invalid_calls": len(remaining),
        "recoveries": details,
        "remaining": remaining,
        "dry_run": args.dry_run,
    }

    report_path = args.report or (args.root / "reparse_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Files scanned: {len(files)}")
    print(f"Changed files: {changed_files}")
    print(f"Recovered calls: {recovered_calls}")
    print(f"Remaining invalid calls: {len(remaining)}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
