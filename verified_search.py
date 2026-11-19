#!/usr/bin/env python3
"""Run symmetry-grounded, RR-verified search over Qwen candidate pools.

Expected input: aggregate.json files created by generate_candidates_qwen35.py.
The script generates independent tool/evaluation RR sets, builds a
consensus-diversity seed set, and performs budgeted counterfactual swap repair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from rr_tools import (  # noqa: E402
    GraphView,
    build_node_rr_bitsets,
    coverage_jaccard,
    covered_bitset,
    estimate_spread,
    leave_one_out_rr_loss,
    load_or_create_rr_sets,
    load_processed_graph,
    marginal_rr_gain,
    swap_rr_delta,
    union_without_node,
)

LOGGER = logging.getLogger("verified_search")


@dataclass
class LLMControllerConfig:
    model_name: str
    torch_dtype: str
    device_map: str
    temperature: float
    top_p: float
    max_new_tokens: int
    local_files_only: bool
    trust_remote_code: bool
    enable_thinking: bool


class SwapController:
    name = "base"

    def propose(
        self,
        *,
        state: dict[str, Any],
        weak_nodes: Sequence[dict[str, Any]],
        additions: Sequence[dict[str, Any]],
        max_proposals: int,
        seed: int,
    ) -> list[tuple[int, int]]:
        raise NotImplementedError


class RandomSwapController(SwapController):
    name = "random"

    def propose(self, *, state, weak_nodes, additions, max_proposals, seed):
        del state
        pairs = [
            (int(w["node"]), int(a["node"]))
            for w in weak_nodes
            for a in additions
            if int(w["node"]) != int(a["node"])
        ]
        rng = random.Random(seed)
        rng.shuffle(pairs)
        return pairs[:max_proposals]


class ProxySwapController(SwapController):
    name = "proxy"

    def propose(self, *, state, weak_nodes, additions, max_proposals, seed):
        del state, seed
        pairs: list[tuple[float, int, int]] = []
        for weak in weak_nodes:
            u = int(weak["node"])
            unique_loss = float(weak["leave_one_out_rr_loss"])
            weak_overlap = float(weak["max_overlap_with_other_seeds"])
            for addition in additions:
                v = int(addition["node"])
                gain = float(addition["marginal_rr_gain"])
                consensus = float(addition["consensus_score_normalized"])
                overlap = float(addition["max_overlap_with_current_seeds"])
                # Proxy only: it does not evaluate the exact swap coverage.
                score = (
                    gain
                    - unique_loss
                    + 0.25 * consensus * max(gain + unique_loss, 1.0)
                    + 0.10 * weak_overlap * max(unique_loss, 1.0)
                    - 0.20 * overlap * max(gain, 1.0)
                )
                pairs.append((score, u, v))
        pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
        return [(u, v) for _, u, v in pairs[:max_proposals]]


class ExhaustiveSwapController(SwapController):
    name = "exhaustive"

    def propose(self, *, state, weak_nodes, additions, max_proposals, seed):
        del state, max_proposals, seed
        return [
            (int(w["node"]), int(a["node"]))
            for w in weak_nodes
            for a in additions
            if int(w["node"]) != int(a["node"])
        ]


class LLMSwapController(SwapController):
    name = "llm"

    def __init__(self, config: LLMControllerConfig) -> None:
        try:
            from generate_candidates_qwen35 import HuggingFaceBackend
        except ImportError as exc:
            raise RuntimeError(
                "LLM controller requires generate_candidates_qwen35.py in the "
                "same VerifiedSearch directory."
            ) from exc
        self.backend = HuggingFaceBackend(
            model_name=config.model_name,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            max_input_tokens=None,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
            enable_thinking=config.enable_thinking,
        )

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I)

    @classmethod
    def _parse_swaps(
        cls,
        text: str,
        weak_ids: set[int],
        addition_ids: set[int],
        max_proposals: int,
    ) -> list[tuple[int, int]]:
        cleaned = cls._strip_thinking(text).strip()
        candidates: list[str] = [cleaned]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if 0 <= start < end:
            candidates.append(cleaned[start : end + 1])

        payload: dict[str, Any] | None = None
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    payload = obj
                    break
            except json.JSONDecodeError:
                continue
        if payload is None:
            return []

        raw = payload.get("proposed_swaps", payload.get("swaps", []))
        if not isinstance(raw, list):
            return []

        swaps: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                u = int(item.get("remove"))
                v = int(item.get("add"))
            except (TypeError, ValueError):
                continue
            pair = (u, v)
            if u not in weak_ids or v not in addition_ids or pair in seen:
                continue
            seen.add(pair)
            swaps.append(pair)
            if len(swaps) >= max_proposals:
                break
        return swaps

    def propose(self, *, state, weak_nodes, additions, max_proposals, seed):
        prompt = (
            "You are controlling a budget-limited local search for influence "
            "maximization. Choose counterfactual swaps only from the supplied "
            "weak seeds and replacement candidates. Prefer removing redundant "
            "seeds and adding candidates with high marginal RR coverage, high "
            "permutation consensus, and low overlap.\n\n"
            f"Search state:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Weak seeds:\n{json.dumps(list(weak_nodes), ensure_ascii=False, indent=2)}\n\n"
            f"Replacement candidates:\n{json.dumps(list(additions), ensure_ascii=False, indent=2)}\n\n"
            f"Return at most {max_proposals} swaps as valid JSON only:\n"
            '{"proposed_swaps":[{"remove":31,"add":77,'
            '"reason":["remove_redundancy","new_coverage"]}]}'
        )
        result = self.backend.generate(
            prompt,
            seed=seed,
            permutation={},
            list_length=max_proposals,
        )
        swaps = self._parse_swaps(
            result.text,
            weak_ids={int(x["node"]) for x in weak_nodes},
            addition_ids={int(x["node"]) for x in additions},
            max_proposals=max_proposals,
        )
        state["llm_controller_metadata"] = result.metadata
        state["llm_raw_output"] = result.text
        return swaps


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *map(str, parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def resolve_graph_path(raw_path: str, candidate_root: Path) -> Path:
    path = Path(raw_path)
    candidates = [
        path,
        Path.cwd() / path,
        PROJECT_ROOT / path,
        SCRIPT_DIR / path,
        candidate_root / path,
        SCRIPT_DIR / "processed_graphs" / path,
    ]
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve graph_file from aggregate.json. Tried:\n  - "
        + "\n  - ".join(str(x.expanduser().resolve()) for x in candidates)
    )


def candidate_score_map(aggregate: dict[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in aggregate.get("candidate_pool", []):
        try:
            result[int(item["internal_node"])] = float(item["consensus_score"])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def max_coverage_overlap(
    node: int,
    selected: Sequence[int],
    bitsets: Sequence[int],
    *,
    exclude_self: bool = False,
) -> float:
    values = []
    skipped = False
    for other in selected:
        if exclude_self and int(other) == int(node) and not skipped:
            skipped = True
            continue
        values.append(coverage_jaccard(bitsets[int(node)], bitsets[int(other)]))
    return max(values, default=0.0)


def build_consensus_diversity_initialization(
    *,
    candidate_pool: Sequence[int],
    consensus_scores: dict[int, float],
    budget: int,
    bitsets: Sequence[int],
) -> tuple[list[int], list[dict[str, Any]]]:
    if budget > len(candidate_pool):
        raise ValueError(
            f"Budget {budget} exceeds candidate pool size {len(candidate_pool)}."
        )
    max_score = max((consensus_scores.get(int(v), 0.0) for v in candidate_pool), default=1.0)
    if max_score <= 0.0:
        max_score = 1.0

    selected: list[int] = []
    trace: list[dict[str, Any]] = []
    covered = 0

    while len(selected) < budget:
        ranked: list[tuple[float, int, int, float, float]] = []
        for raw_node in candidate_pool:
            node = int(raw_node)
            if node in selected:
                continue
            consensus_norm = consensus_scores.get(node, 0.0) / max_score
            overlap = max_coverage_overlap(node, selected, bitsets)
            novelty = 1.0 - overlap
            score = consensus_norm * novelty
            marginal = marginal_rr_gain(node, covered, bitsets)
            ranked.append((score, marginal, -node, overlap, consensus_norm))

        if not ranked:
            raise RuntimeError("Candidate pool exhausted before filling the budget.")
        ranked.sort(reverse=True)
        score, marginal, negative_node, overlap, consensus_norm = ranked[0]
        node = -negative_node
        selected.append(node)
        covered |= bitsets[node]
        trace.append(
            {
                "step": len(selected),
                "selected_node": node,
                "selection_score": score,
                "consensus_score_normalized": consensus_norm,
                "max_overlap_with_selected": overlap,
                "marginal_rr_gain": marginal,
                "covered_rr_sets_after": covered.bit_count(),
            }
        )

    return selected, trace


def weak_seed_statistics(
    seed_set: Sequence[int],
    bitsets: Sequence[int],
    num_weak: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in seed_set:
        rows.append(
            {
                "node": int(node),
                "leave_one_out_rr_loss": leave_one_out_rr_loss(seed_set, node, bitsets),
                "max_overlap_with_other_seeds": max_coverage_overlap(
                    int(node), seed_set, bitsets, exclude_self=True
                ),
            }
        )
    rows.sort(
        key=lambda x: (
            x["leave_one_out_rr_loss"],
            -x["max_overlap_with_other_seeds"],
            x["node"],
        )
    )
    return rows[: min(num_weak, len(rows))]


def addition_statistics(
    *,
    seed_set: Sequence[int],
    candidate_pool: Sequence[int],
    consensus_scores: dict[int, float],
    bitsets: Sequence[int],
    num_additions: int,
    num_nodes: int,
    num_rr_sets: int,
    graph: GraphView,
) -> list[dict[str, Any]]:
    selected = set(map(int, seed_set))
    covered = covered_bitset(seed_set, bitsets)
    max_score = max(consensus_scores.values(), default=1.0)
    if max_score <= 0:
        max_score = 1.0

    rows: list[dict[str, Any]] = []
    for raw_node in candidate_pool:
        node = int(raw_node)
        if node in selected:
            continue
        rr_gain = marginal_rr_gain(node, covered, bitsets)
        rows.append(
            {
                "node": node,
                "marginal_rr_gain": rr_gain,
                "estimated_marginal_spread": num_nodes * rr_gain / num_rr_sets,
                "consensus_score": consensus_scores.get(node, 0.0),
                "consensus_score_normalized": consensus_scores.get(node, 0.0) / max_score,
                "max_overlap_with_current_seeds": max_coverage_overlap(
                    node, seed_set, bitsets
                ),
                "out_degree": graph.out_degree[node],
                "in_degree": graph.in_degree[node],
            }
        )
    rows.sort(
        key=lambda x: (
            -x["marginal_rr_gain"],
            -x["consensus_score"],
            x["max_overlap_with_current_seeds"],
            x["node"],
        )
    )
    return rows[: min(num_additions, len(rows))]


def run_verified_repair(
    *,
    initial_seed_set: Sequence[int],
    candidate_pool: Sequence[int],
    consensus_scores: dict[int, float],
    graph: GraphView,
    bitsets: Sequence[int],
    num_rr_sets: int,
    controller: SwapController,
    rounds: int,
    num_weak: int,
    num_additions: int,
    max_proposals: int,
    min_rr_gain: int,
    base_seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    current = list(map(int, initial_seed_set))
    history: list[dict[str, Any]] = []

    for round_index in range(rounds):
        current_covered = covered_bitset(current, bitsets)
        weak = weak_seed_statistics(current, bitsets, num_weak)
        additions = addition_statistics(
            seed_set=current,
            candidate_pool=candidate_pool,
            consensus_scores=consensus_scores,
            bitsets=bitsets,
            num_additions=num_additions,
            num_nodes=graph.num_nodes,
            num_rr_sets=num_rr_sets,
            graph=graph,
        )
        state: dict[str, Any] = {
            "round": round_index + 1,
            "current_seed_set": current,
            "current_covered_rr_sets": current_covered.bit_count(),
            "estimated_spread": graph.num_nodes
            * current_covered.bit_count()
            / num_rr_sets,
            "remaining_rounds": rounds - round_index,
            "controller": controller.name,
        }
        proposal_seed = stable_seed(base_seed, graph.name, round_index, controller.name)
        proposals = controller.propose(
            state=state,
            weak_nodes=weak,
            additions=additions,
            max_proposals=max_proposals,
            seed=proposal_seed,
        )

        # Deduplicate and validate before verification.
        weak_ids = {int(x["node"]) for x in weak}
        addition_ids = {int(x["node"]) for x in additions}
        clean_proposals: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for u, v in proposals:
            pair = (int(u), int(v))
            if pair in seen or pair[0] not in weak_ids or pair[1] not in addition_ids:
                continue
            seen.add(pair)
            clean_proposals.append(pair)

        evaluations: list[dict[str, Any]] = []
        for u, v in clean_proposals:
            delta, new_covered = swap_rr_delta(current, u, v, bitsets)
            without = union_without_node(current, u, bitsets)
            evaluations.append(
                {
                    "remove": u,
                    "add": v,
                    "delta_rr_sets": delta,
                    "delta_estimated_spread": graph.num_nodes * delta / num_rr_sets,
                    "covered_rr_sets_after": new_covered.bit_count(),
                    "rr_sets_lost_before_add": current_covered.bit_count()
                    - without.bit_count(),
                    "new_rr_sets_added_after_remove": (
                        bitsets[v] & ~without
                    ).bit_count(),
                }
            )
        evaluations.sort(
            key=lambda x: (
                -x["delta_rr_sets"],
                x["remove"],
                x["add"],
            )
        )

        accepted = None
        if evaluations and evaluations[0]["delta_rr_sets"] > min_rr_gain:
            accepted = evaluations[0]
            current = [
                accepted["add"] if node == accepted["remove"] else node
                for node in current
            ]
            if len(set(current)) != len(current):
                raise RuntimeError("Accepted swap created duplicate seed nodes.")

        history.append(
            {
                **state,
                "weak_seeds": weak,
                "replacement_candidates": additions,
                "proposed_swaps": [
                    {"remove": u, "add": v} for u, v in clean_proposals
                ],
                "verified_swaps": evaluations,
                "accepted_swap": accepted,
                "seed_set_after_round": current,
            }
        )
        if accepted is None:
            break

    return current, history


def method_metrics(
    seed_set: Sequence[int],
    *,
    tool_bitsets: Sequence[int],
    eval_bitsets: Sequence[int],
    graph: GraphView,
    tool_rr_count: int,
    eval_rr_count: int,
) -> dict[str, Any]:
    return {
        "seed_set": list(map(int, seed_set)),
        "tool_estimate": estimate_spread(
            seed_set,
            node_rr_bitsets=tool_bitsets,
            num_nodes=graph.num_nodes,
            num_rr_sets=tool_rr_count,
        ).to_dict(),
        "evaluation_estimate": estimate_spread(
            seed_set,
            node_rr_bitsets=eval_bitsets,
            num_nodes=graph.num_nodes,
            num_rr_sets=eval_rr_count,
        ).to_dict(),
    }


def discover_aggregates(root: Path) -> list[Path]:
    return sorted(root.rglob("aggregate.json"))


def build_controller(args: argparse.Namespace) -> SwapController:
    if args.controller == "random":
        return RandomSwapController()
    if args.controller == "proxy":
        return ProxySwapController()
    if args.controller == "exhaustive":
        return ExhaustiveSwapController()
    if args.controller == "llm":
        return LLMSwapController(
            LLMControllerConfig(
                model_name=args.model_name,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
                enable_thinking=args.enable_thinking,
            )
        )
    raise ValueError(f"Unknown controller: {args.controller}")


def process_aggregate(
    aggregate_path: Path,
    *,
    args: argparse.Namespace,
    controller: SwapController,
) -> tuple[dict[str, Any], Path]:
    aggregate = load_json(aggregate_path)
    graph_name = str(aggregate["graph_name"])
    dataset_type = str(aggregate["dataset_type"])
    diffusion = str(aggregate["diffusion"])

    graph_path = resolve_graph_path(str(aggregate["graph_file"]), args.candidate_root)
    graph_json, graph = load_processed_graph(graph_path)
    del graph_json

    relative_parent = aggregate_path.parent.relative_to(args.candidate_root)
    output_path = args.output_dir / relative_parent / f"verified_{controller.name}.json"
    if output_path.exists() and not args.overwrite:
        LOGGER.info("Skipping existing result: %s", output_path)
        return load_json(output_path), output_path

    cache_dir = args.rr_cache_dir / dataset_type / graph_name / diffusion
    tool_seed = stable_seed(args.base_seed, graph_name, diffusion, "tool_rr")
    eval_seed = stable_seed(args.base_seed, graph_name, diffusion, "eval_rr")
    tool_cache = cache_dir / f"tool_rr_{args.tool_rr_sets}_seed{tool_seed}.pkl.gz"
    eval_cache = cache_dir / f"eval_rr_{args.eval_rr_sets}_seed{eval_seed}.pkl.gz"

    LOGGER.info(
        "Generating/loading RR sets graph=%s diffusion=%s tool=%d eval=%d",
        graph_name,
        diffusion,
        args.tool_rr_sets,
        args.eval_rr_sets,
    )
    tool_rr = load_or_create_rr_sets(
        tool_cache,
        graph=graph,
        diffusion_name=diffusion,
        num_rr_sets=args.tool_rr_sets,
        seed=tool_seed,
        overwrite=args.overwrite_rr_cache,
    )
    eval_rr = load_or_create_rr_sets(
        eval_cache,
        graph=graph,
        diffusion_name=diffusion,
        num_rr_sets=args.eval_rr_sets,
        seed=eval_seed,
        overwrite=args.overwrite_rr_cache,
    )
    tool_bitsets = build_node_rr_bitsets(tool_rr, graph.num_nodes)
    eval_bitsets = build_node_rr_bitsets(eval_rr, graph.num_nodes)

    candidate_pool = [int(x) for x in aggregate.get("candidate_pool_ids", [])]
    consensus_scores = candidate_score_map(aggregate)
    if not candidate_pool:
        raise ValueError(f"No candidate_pool_ids in {aggregate_path}")

    results_by_budget: dict[str, Any] = {}
    for budget in map(int, aggregate.get("budgets", [])):
        if budget > len(candidate_pool):
            LOGGER.warning(
                "Skipping graph=%s budget=%d: candidate pool has only %d nodes.",
                graph_name,
                budget,
                len(candidate_pool),
            )
            continue

        baseline = aggregate.get("consensus_seed_baselines", {}).get(str(budget))
        if not baseline:
            baseline = candidate_pool[:budget]
        baseline = [int(x) for x in baseline]

        initial, initialization_trace = build_consensus_diversity_initialization(
            candidate_pool=candidate_pool,
            consensus_scores=consensus_scores,
            budget=budget,
            bitsets=tool_bitsets,
        )
        final, repair_history = run_verified_repair(
            initial_seed_set=initial,
            candidate_pool=candidate_pool,
            consensus_scores=consensus_scores,
            graph=graph,
            bitsets=tool_bitsets,
            num_rr_sets=args.tool_rr_sets,
            controller=controller,
            rounds=args.rounds,
            num_weak=args.weak_seeds,
            num_additions=args.addition_candidates,
            max_proposals=args.proposals,
            min_rr_gain=args.min_rr_gain,
            base_seed=stable_seed(args.base_seed, graph_name, diffusion, budget),
        )

        baseline_metrics = method_metrics(
            baseline,
            tool_bitsets=tool_bitsets,
            eval_bitsets=eval_bitsets,
            graph=graph,
            tool_rr_count=args.tool_rr_sets,
            eval_rr_count=args.eval_rr_sets,
        )
        initial_metrics = method_metrics(
            initial,
            tool_bitsets=tool_bitsets,
            eval_bitsets=eval_bitsets,
            graph=graph,
            tool_rr_count=args.tool_rr_sets,
            eval_rr_count=args.eval_rr_sets,
        )
        final_metrics = method_metrics(
            final,
            tool_bitsets=tool_bitsets,
            eval_bitsets=eval_bitsets,
            graph=graph,
            tool_rr_count=args.tool_rr_sets,
            eval_rr_count=args.eval_rr_sets,
        )
        base_eval = baseline_metrics["evaluation_estimate"]["spread"]
        init_eval = initial_metrics["evaluation_estimate"]["spread"]
        final_eval = final_metrics["evaluation_estimate"]["spread"]

        results_by_budget[str(budget)] = {
            "consensus_baseline": baseline_metrics,
            "consensus_diversity_initialization": initial_metrics,
            "verified_search": final_metrics,
            "initialization_trace": initialization_trace,
            "repair_history": repair_history,
            "evaluation_gain_over_consensus": final_eval - base_eval,
            "evaluation_gain_over_initialization": final_eval - init_eval,
            "num_accepted_swaps": sum(
                1 for item in repair_history if item["accepted_swap"] is not None
            ),
            "num_verified_swaps": sum(
                len(item["verified_swaps"]) for item in repair_history
            ),
        }

    payload = {
        "version": 1,
        "graph_name": graph_name,
        "dataset_type": dataset_type,
        "graph_file": str(graph_path),
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "diffusion": diffusion,
        "candidate_aggregate_file": str(aggregate_path),
        "candidate_model": aggregate.get("model_name"),
        "candidate_pool_size": len(candidate_pool),
        "controller": controller.name,
        "configuration": {
            "tool_rr_sets": args.tool_rr_sets,
            "eval_rr_sets": args.eval_rr_sets,
            "tool_rr_seed": tool_seed,
            "eval_rr_seed": eval_seed,
            "rounds": args.rounds,
            "weak_seeds": args.weak_seeds,
            "addition_candidates": args.addition_candidates,
            "proposals": args.proposals,
            "min_rr_gain": args.min_rr_gain,
            "base_seed": args.base_seed,
        },
        "budgets": results_by_budget,
    }
    save_json(payload, output_path)
    return payload, output_path


def write_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "graph_name",
        "dataset_type",
        "diffusion",
        "budget",
        "controller",
        "consensus_eval_spread",
        "initial_eval_spread",
        "final_eval_spread",
        "gain_over_consensus",
        "gain_over_initialization",
        "accepted_swaps",
        "verified_swaps",
        "result_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RR-set initialization and verified counterfactual search."
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=SCRIPT_DIR / "candidate_outputs_qwen35_9b",
        help="Root containing recursively discovered aggregate.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "verified_search_outputs",
    )
    parser.add_argument(
        "--rr-cache-dir",
        type=Path,
        default=SCRIPT_DIR / "rr_cache",
    )
    parser.add_argument(
        "--controller",
        choices=("proxy", "random", "exhaustive", "llm"),
        default="proxy",
    )
    parser.add_argument("--tool-rr-sets", type=int, default=500)
    parser.add_argument("--eval-rr-sets", type=int, default=10000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--weak-seeds", type=int, default=3)
    parser.add_argument("--addition-candidates", type=int, default=10)
    parser.add_argument("--proposals", type=int, default=4)
    parser.add_argument(
        "--min-rr-gain",
        type=int,
        default=0,
        help="Accept only swaps with delta_rr_sets greater than this value.",
    )
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--graphs", nargs="*", default=None)
    parser.add_argument(
        "--dataset-types", nargs="*", choices=("synthetic", "real"), default=None
    )
    parser.add_argument("--diffusions", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-rr-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # LLM controller options.
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true")
    thinking.add_argument("--no-enable-thinking", action="store_false", dest="enable_thinking")
    parser.set_defaults(enable_thinking=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args.candidate_root = args.candidate_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.rr_cache_dir = args.rr_cache_dir.expanduser().resolve()

    if args.tool_rr_sets <= 0 or args.eval_rr_sets <= 0:
        raise ValueError("RR-set counts must be positive.")
    if args.rounds < 0 or args.weak_seeds <= 0 or args.addition_candidates <= 0:
        raise ValueError("Invalid search configuration.")
    if args.proposals <= 0:
        raise ValueError("--proposals must be positive.")

    aggregate_paths = discover_aggregates(args.candidate_root)
    selected: list[Path] = []
    for path in aggregate_paths:
        try:
            aggregate = load_json(path)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable aggregate %s: %s", path, exc)
            continue
        if args.graphs and str(aggregate.get("graph_name")) not in set(args.graphs):
            continue
        if args.dataset_types and str(aggregate.get("dataset_type")) not in set(args.dataset_types):
            continue
        if args.diffusions and str(aggregate.get("diffusion")) not in set(args.diffusions):
            continue
        selected.append(path)

    if not selected:
        LOGGER.error("No aggregate.json files matched under %s", args.candidate_root)
        return 1

    LOGGER.info("Selected %d graph/diffusion job(s).", len(selected))
    for path in selected:
        LOGGER.info("  %s", path)
    if args.dry_run:
        return 0

    controller = build_controller(args)
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    for aggregate_path in selected:
        try:
            payload, output_path = process_aggregate(
                aggregate_path,
                args=args,
                controller=controller,
            )
            for budget, item in payload["budgets"].items():
                summary_rows.append(
                    {
                        "graph_name": payload["graph_name"],
                        "dataset_type": payload["dataset_type"],
                        "diffusion": payload["diffusion"],
                        "budget": int(budget),
                        "controller": payload["controller"],
                        "consensus_eval_spread": item["consensus_baseline"]["evaluation_estimate"]["spread"],
                        "initial_eval_spread": item["consensus_diversity_initialization"]["evaluation_estimate"]["spread"],
                        "final_eval_spread": item["verified_search"]["evaluation_estimate"]["spread"],
                        "gain_over_consensus": item["evaluation_gain_over_consensus"],
                        "gain_over_initialization": item["evaluation_gain_over_initialization"],
                        "accepted_swaps": item["num_accepted_swaps"],
                        "verified_swaps": item["num_verified_swaps"],
                        "result_file": str(output_path),
                    }
                )
            LOGGER.info(
                "Completed graph=%s diffusion=%s -> %s",
                payload["graph_name"],
                payload["diffusion"],
                output_path,
            )
        except Exception as exc:
            LOGGER.exception("Failed aggregate=%s", aggregate_path)
            failures.append({"aggregate": str(aggregate_path), "error": str(exc)})

    summary_path = args.output_dir / f"summary_{controller.name}.csv"
    write_summary(summary_rows, summary_path)
    save_json(
        {
            "controller": controller.name,
            "candidate_root": str(args.candidate_root),
            "completed_jobs": len({(r['graph_name'], r['diffusion']) for r in summary_rows}),
            "budget_rows": len(summary_rows),
            "failures": failures,
            "elapsed_seconds": time.perf_counter() - started,
            "summary_csv": str(summary_path),
        },
        args.output_dir / f"index_{controller.name}.json",
    )
    LOGGER.info(
        "Finished: %d budget result(s), %d failure(s). Summary: %s",
        len(summary_rows),
        len(failures),
        summary_path,
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
