#!/usr/bin/env python3
"""Run fixed-verification-budget RR search over Qwen candidate pools.

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
    """Select proposal IDs from an explicitly enumerated swap table."""

    name = "base_fixed"

    def propose(
        self,
        *,
        state: dict[str, Any],
        candidate_swaps: Sequence[dict[str, Any]],
        max_proposals: int,
        seed: int,
    ) -> list[int]:
        raise NotImplementedError


class RandomSwapController(SwapController):
    name = "random_fixed"

    def propose(self, *, state, candidate_swaps, max_proposals, seed):
        del state
        ids = [int(item["id"]) for item in candidate_swaps]
        rng = random.Random(seed)
        rng.shuffle(ids)
        return ids[:max_proposals]


class ProxySwapController(SwapController):
    name = "proxy_fixed"

    def propose(self, *, state, candidate_swaps, max_proposals, seed):
        del state, seed
        ranked = sorted(
            candidate_swaps,
            key=lambda item: (
                -float(item["proxy_score"]),
                int(item["remove"]),
                int(item["add"]),
            ),
        )
        return [int(item["id"]) for item in ranked[:max_proposals]]


class ExhaustiveSwapController(SwapController):
    name = "exhaustive_fixed"

    def propose(self, *, state, candidate_swaps, max_proposals, seed):
        del state, max_proposals, seed
        return [int(item["id"]) for item in candidate_swaps]


class LLMSwapController(SwapController):
    """Qwen controller that chooses IDs rather than emitting node IDs directly."""

    def __init__(
        self,
        config: LLMControllerConfig,
        *,
        proposal_retries: int,
        fill_strategy: str,
    ) -> None:
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
        self.proposal_retries = max(0, int(proposal_retries))
        self.fill_strategy = str(fill_strategy)
        self.name = f"llm_fixed_{self.fill_strategy}fill"

    @staticmethod
    def _strip_thinking(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I)
        if "<think>" in cleaned.lower():
            cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.I)
        return cleaned.strip()

    @classmethod
    def _parse_proposal_ids(
        cls,
        text: str,
        *,
        allowed_ids: set[int],
        max_proposals: int,
    ) -> list[int]:
        cleaned = cls._strip_thinking(text)
        fragments = [cleaned]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if 0 <= start < end:
            fragments.append(cleaned[start : end + 1])

        raw_ids: list[Any] = []
        for fragment in fragments:
            try:
                payload = json.loads(fragment)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                value = payload.get("proposal_ids", payload.get("ids", []))
                if isinstance(value, list):
                    raw_ids = value
                    break
            elif isinstance(payload, list):
                raw_ids = payload
                break

        if not raw_ids:
            match = re.search(
                r'"?(?:proposal_ids|ids)"?\s*:\s*\[([^\]]*)\]',
                cleaned,
                flags=re.I | re.S,
            )
            if match:
                raw_ids = re.findall(r"-?\d+", match.group(1))

        result: list[int] = []
        seen: set[int] = set()
        for raw in raw_ids:
            try:
                proposal_id = int(raw)
            except (TypeError, ValueError):
                continue
            if proposal_id not in allowed_ids or proposal_id in seen:
                continue
            seen.add(proposal_id)
            result.append(proposal_id)
            if len(result) >= max_proposals:
                break
        return result

    def propose(self, *, state, candidate_swaps, max_proposals, seed):
        selected: list[int] = []
        attempts: list[dict[str, Any]] = []

        for attempt_index in range(self.proposal_retries + 1):
            selected_set = set(selected)
            remaining = [
                item for item in candidate_swaps if int(item["id"]) not in selected_set
            ]
            needed = max_proposals - len(selected)
            if needed <= 0 or not remaining:
                break

            retry_note = ""
            if attempt_index > 0:
                retry_note = (
                    f"\nPrevious attempts yielded only {len(selected)} valid unique IDs. "
                    f"Select exactly {needed} additional IDs from the remaining table."
                )

            prompt = (
                "/no_think\n"
                "You are selecting counterfactual swaps for a budget-limited "
                "influence-maximization local search. Every legal swap has already "
                "been assigned an integer ID. Select only IDs from the supplied table. "
                "Prefer high expected coverage gain, removal of redundant seeds, "
                "high permutation consensus, and low overlap. Do not invent node IDs.\n\n"
                f"Search state:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
                f"Candidate swap table:\n{json.dumps(remaining, ensure_ascii=False, indent=2)}\n\n"
                f"Return exactly {needed} distinct IDs as valid JSON only:\n"
                '{"proposal_ids":[1,7,13,20]}'
                f"{retry_note}"
            )
            result = self.backend.generate(
                prompt,
                seed=stable_seed(seed, attempt_index),
                permutation={},
                list_length=needed,
            )
            parsed = self._parse_proposal_ids(
                result.text,
                allowed_ids={int(item["id"]) for item in remaining},
                max_proposals=needed,
            )
            for proposal_id in parsed:
                if proposal_id not in selected:
                    selected.append(proposal_id)
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "needed": needed,
                    "parsed_ids": parsed,
                    "raw_output": result.text,
                    "backend_metadata": result.metadata,
                }
            )

        state["llm_controller_attempts"] = attempts
        state["llm_valid_proposal_count"] = len(selected)
        return selected[:max_proposals]


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



def _proxy_score(weak: dict[str, Any], addition: dict[str, Any]) -> float:
    unique_loss = float(weak["leave_one_out_rr_loss"])
    weak_overlap = float(weak["max_overlap_with_other_seeds"])
    gain = float(addition["marginal_rr_gain"])
    consensus = float(addition["consensus_score_normalized"])
    overlap = float(addition["max_overlap_with_current_seeds"])
    return (
        gain
        - unique_loss
        + 0.25 * consensus * max(gain + unique_loss, 1.0)
        + 0.10 * weak_overlap * max(unique_loss, 1.0)
        - 0.20 * overlap * max(gain, 1.0)
    )


def _build_candidate_swap_table(
    *,
    weak_nodes: Sequence[dict[str, Any]],
    additions: Sequence[dict[str, Any]],
    state_signature: tuple[int, ...],
    tested_state_pairs: set[tuple[tuple[int, ...], int, int]],
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for weak in weak_nodes:
        u = int(weak["node"])
        for addition in additions:
            v = int(addition["node"])
            key = (state_signature, u, v)
            if u == v or key in tested_state_pairs:
                continue
            table.append(
                {
                    "id": len(table) + 1,
                    "remove": u,
                    "add": v,
                    "remove_unique_rr_loss": int(weak["leave_one_out_rr_loss"]),
                    "remove_max_overlap": float(
                        weak["max_overlap_with_other_seeds"]
                    ),
                    "add_marginal_rr_gain": int(addition["marginal_rr_gain"]),
                    "add_consensus_normalized": float(
                        addition["consensus_score_normalized"]
                    ),
                    "add_max_overlap": float(
                        addition["max_overlap_with_current_seeds"]
                    ),
                    "add_out_degree": int(addition["out_degree"]),
                    "add_in_degree": int(addition["in_degree"]),
                    "proxy_score": _proxy_score(weak, addition),
                }
            )
    return table


def _fill_proposal_ids(
    *,
    selected_ids: Sequence[int],
    candidate_swaps: Sequence[dict[str, Any]],
    target_count: int,
    strategy: str,
    seed: int,
) -> list[int]:
    result = list(dict.fromkeys(map(int, selected_ids)))
    selected_set = set(result)
    remaining = [
        item for item in candidate_swaps if int(item["id"]) not in selected_set
    ]
    needed = max(0, target_count - len(result))
    if needed == 0 or strategy == "none":
        return result[:target_count]

    if strategy == "proxy":
        remaining.sort(
            key=lambda item: (
                -float(item["proxy_score"]),
                int(item["remove"]),
                int(item["add"]),
            )
        )
    elif strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(remaining)
    else:
        raise ValueError(f"Unknown fill strategy: {strategy}")

    result.extend(int(item["id"]) for item in remaining[:needed])
    return result[:target_count]


def run_verified_repair(
    *,
    initial_seed_set: Sequence[int],
    candidate_pool: Sequence[int],
    consensus_scores: dict[int, float],
    graph: GraphView,
    bitsets: Sequence[int],
    num_rr_sets: int,
    controller: SwapController,
    verification_budget: int,
    batch_size: int,
    max_accepted_swaps: int,
    num_weak: int,
    num_additions: int,
    min_rr_gain: int,
    base_seed: int,
    llm_fill_strategy: str,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    current = list(map(int, initial_seed_set))
    history: list[dict[str, Any]] = []
    tested_state_pairs: set[tuple[tuple[int, ...], int, int]] = set()
    verified_total = 0
    accepted_total = 0
    batch_index = 0
    llm_selected_total = 0
    fill_selected_total = 0
    stop_reason = "verification_budget_exhausted"

    while verified_total < verification_budget:
        if accepted_total >= max_accepted_swaps:
            stop_reason = "max_accepted_swaps_reached"
            break

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
        state_signature = tuple(sorted(map(int, current)))
        candidate_swaps = _build_candidate_swap_table(
            weak_nodes=weak,
            additions=additions,
            state_signature=state_signature,
            tested_state_pairs=tested_state_pairs,
        )
        if not candidate_swaps:
            stop_reason = "candidate_swaps_exhausted"
            break

        batch_index += 1
        remaining_budget = verification_budget - verified_total
        target_count = min(batch_size, remaining_budget, len(candidate_swaps))
        state: dict[str, Any] = {
            "batch": batch_index,
            "current_seed_set": current,
            "current_covered_rr_sets": current_covered.bit_count(),
            "estimated_spread": graph.num_nodes
            * current_covered.bit_count()
            / num_rr_sets,
            "verification_budget": verification_budget,
            "verified_so_far": verified_total,
            "remaining_verifications": remaining_budget,
            "accepted_swaps_so_far": accepted_total,
            "max_accepted_swaps": max_accepted_swaps,
            "controller": controller.name,
        }
        proposal_seed = stable_seed(
            base_seed,
            graph.name,
            state_signature,
            batch_index,
            controller.name,
        )
        raw_ids = controller.propose(
            state=state,
            candidate_swaps=candidate_swaps,
            max_proposals=target_count,
            seed=proposal_seed,
        )

        valid_ids: list[int] = []
        allowed = {int(item["id"]) for item in candidate_swaps}
        for raw_id in raw_ids:
            proposal_id = int(raw_id)
            if proposal_id in allowed and proposal_id not in valid_ids:
                valid_ids.append(proposal_id)
            if len(valid_ids) >= target_count:
                break

        llm_count = len(valid_ids) if controller.name.startswith("llm_fixed") else 0
        selected_ids = valid_ids
        fill_strategy = (
            llm_fill_strategy if controller.name.startswith("llm_fixed") else "none"
        )
        if len(selected_ids) < target_count:
            selected_ids = _fill_proposal_ids(
                selected_ids=selected_ids,
                candidate_swaps=candidate_swaps,
                target_count=target_count,
                strategy=fill_strategy,
                seed=stable_seed(proposal_seed, "fill"),
            )
        fill_count = (
            max(0, len(selected_ids) - llm_count)
            if controller.name.startswith("llm_fixed")
            else 0
        )
        llm_selected_total += llm_count
        fill_selected_total += fill_count

        if not selected_ids:
            stop_reason = "no_valid_proposals"
            history.append(
                {
                    **state,
                    "weak_seeds": weak,
                    "replacement_candidates": additions,
                    "candidate_swaps": candidate_swaps,
                    "selected_proposal_ids": [],
                    "llm_selected_count": llm_count,
                    "fill_selected_count": fill_count,
                    "verified_swaps": [],
                    "accepted_swap": None,
                    "seed_set_after_batch": current,
                }
            )
            break

        id_to_swap = {int(item["id"]): item for item in candidate_swaps}
        selected_records = [id_to_swap[proposal_id] for proposal_id in selected_ids]
        evaluations: list[dict[str, Any]] = []
        for record in selected_records:
            u = int(record["remove"])
            v = int(record["add"])
            tested_state_pairs.add((state_signature, u, v))
            delta, new_covered = swap_rr_delta(current, u, v, bitsets)
            without = union_without_node(current, u, bitsets)
            evaluations.append(
                {
                    "proposal_id": int(record["id"]),
                    "remove": u,
                    "add": v,
                    "proxy_score": float(record["proxy_score"]),
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
        verified_total += len(evaluations)
        evaluations.sort(
            key=lambda item: (
                -int(item["delta_rr_sets"]),
                int(item["remove"]),
                int(item["add"]),
            )
        )

        accepted = None
        if evaluations and int(evaluations[0]["delta_rr_sets"]) > min_rr_gain:
            accepted = evaluations[0]
            current = [
                int(accepted["add"]) if node == int(accepted["remove"]) else node
                for node in current
            ]
            if len(set(current)) != len(current):
                raise RuntimeError("Accepted swap created duplicate seed nodes.")
            accepted_total += 1

        history.append(
            {
                **state,
                "weak_seeds": weak,
                "replacement_candidates": additions,
                "candidate_swaps": candidate_swaps,
                "selected_proposal_ids": selected_ids,
                "llm_selected_count": llm_count,
                "fill_selected_count": fill_count,
                "verified_swaps": evaluations,
                "accepted_swap": accepted,
                "seed_set_after_batch": current,
                "verified_total_after_batch": verified_total,
                "accepted_total_after_batch": accepted_total,
            }
        )

    diagnostics = {
        "verification_budget": verification_budget,
        "verified_swaps": verified_total,
        "budget_utilization": verified_total / verification_budget
        if verification_budget > 0
        else 0.0,
        "accepted_swaps": accepted_total,
        "proposal_batches": batch_index,
        "llm_selected_proposals": llm_selected_total,
        "fill_selected_proposals": fill_selected_total,
        "stop_reason": stop_reason,
    }
    return current, history, diagnostics


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
    if args.controller == "random_fixed":
        return RandomSwapController()
    if args.controller == "proxy_fixed":
        return ProxySwapController()
    if args.controller == "exhaustive_fixed":
        return ExhaustiveSwapController()
    if args.controller == "llm_fixed":
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
            ),
            proposal_retries=args.llm_proposal_retries,
            fill_strategy=args.llm_fill,
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
        final, repair_history, repair_diagnostics = run_verified_repair(
            initial_seed_set=initial,
            candidate_pool=candidate_pool,
            consensus_scores=consensus_scores,
            graph=graph,
            bitsets=tool_bitsets,
            num_rr_sets=args.tool_rr_sets,
            controller=controller,
            verification_budget=args.verification_budget,
            batch_size=args.batch_size,
            max_accepted_swaps=args.max_accepted_swaps,
            num_weak=args.weak_seeds,
            num_additions=args.addition_candidates,
            min_rr_gain=args.min_rr_gain,
            base_seed=stable_seed(args.base_seed, graph_name, diffusion, budget),
            llm_fill_strategy=args.llm_fill,
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
            "repair_diagnostics": repair_diagnostics,
            "evaluation_gain_over_consensus": final_eval - base_eval,
            "evaluation_gain_over_initialization": final_eval - init_eval,
            "num_accepted_swaps": repair_diagnostics["accepted_swaps"],
            "num_verified_swaps": repair_diagnostics["verified_swaps"],
            "verification_budget_utilization": repair_diagnostics[
                "budget_utilization"
            ],
            "proposal_batches": repair_diagnostics["proposal_batches"],
            "llm_selected_proposals": repair_diagnostics[
                "llm_selected_proposals"
            ],
            "fill_selected_proposals": repair_diagnostics[
                "fill_selected_proposals"
            ],
        }

    payload = {
        "version": 2,
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
            "verification_budget": args.verification_budget,
            "batch_size": args.batch_size,
            "max_accepted_swaps": args.max_accepted_swaps,
            "weak_seeds": args.weak_seeds,
            "addition_candidates": args.addition_candidates,
            "llm_proposal_retries": args.llm_proposal_retries,
            "llm_fill": args.llm_fill,
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
        "verification_budget",
        "budget_utilization",
        "proposal_batches",
        "llm_selected_proposals",
        "fill_selected_proposals",
        "stop_reason",
        "result_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-budget RR-verified counterfactual search."
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
        choices=("proxy_fixed", "random_fixed", "exhaustive_fixed", "llm_fixed"),
        default="proxy_fixed",
    )
    parser.add_argument("--tool-rr-sets", type=int, default=500)
    parser.add_argument("--eval-rr-sets", type=int, default=10000)
    parser.add_argument(
        "--verification-budget",
        type=int,
        default=12,
        help="Maximum number of exact swap evaluations per instance.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of swaps verified together before accepting at most one.",
    )
    parser.add_argument(
        "--max-accepted-swaps",
        type=int,
        default=3,
        help="Maximum number of accepted state-changing swaps.",
    )
    parser.add_argument("--weak-seeds", type=int, default=3)
    parser.add_argument("--addition-candidates", type=int, default=10)
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
    parser.add_argument(
        "--llm-proposal-retries",
        type=int,
        default=2,
        help="Additional attempts to obtain enough valid proposal IDs.",
    )
    parser.add_argument(
        "--llm-fill",
        choices=("random", "proxy", "none"),
        default="random",
        help="How to fill missing LLM proposal IDs for fixed-budget evaluation.",
    )
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
    if args.weak_seeds <= 0 or args.addition_candidates <= 0:
        raise ValueError("Invalid search configuration.")
    if args.verification_budget <= 0:
        raise ValueError("--verification-budget must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_accepted_swaps <= 0:
        raise ValueError("--max-accepted-swaps must be positive.")
    if args.llm_proposal_retries < 0:
        raise ValueError("--llm-proposal-retries cannot be negative.")

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
                        "verification_budget": args.verification_budget,
                        "budget_utilization": item[
                            "verification_budget_utilization"
                        ],
                        "proposal_batches": item["proposal_batches"],
                        "llm_selected_proposals": item[
                            "llm_selected_proposals"
                        ],
                        "fill_selected_proposals": item[
                            "fill_selected_proposals"
                        ],
                        "stop_reason": item["repair_diagnostics"]["stop_reason"],
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
