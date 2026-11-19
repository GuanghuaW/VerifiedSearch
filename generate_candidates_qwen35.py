#!/usr/bin/env python3
"""
Generate permutation-symmetrized LLM candidate rankings for influence maximization.

Expected project layout
-----------------------
IM/
├── sdata/
├── ndata/
└── VerifiedSearch/
    ├── preprocess_graphs.py
    ├── generate_candidates.py
    └── processed_graphs/
        ├── manifest.json
        ├── synthetic/<graph>/...
        └── real/<graph>/...

This is Stage 1 of the verified-search pipeline:
  1. Read each preprocessed graph and its node-ID permutations.
  2. Ask an LLM for one ranked candidate list per permutation.
  3. Map candidates back to the canonical internal node IDs.
  4. Aggregate rankings with a log-discounted consensus score.
  5. Save a candidate pool and permutation-stability statistics.

A deterministic ``degree`` backend is included only for smoke-testing the data
pipeline. Use ``--backend hf`` for the actual LLM experiment.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import logging
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "processed_graphs" / "manifest.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "candidate_outputs"

LOGGER = logging.getLogger("generate_candidates")


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return slug.strip("_") or "unknown"


def resolve_manifest_path(raw_path: str | Path, manifest_path: Path) -> Path:
    """Resolve paths stored by preprocessing without duplicating output roots.

    The preprocessing manifest may contain any of these forms:
      * synthetic/<graph>/graph.json
      * processed_graphs/synthetic/<graph>/graph.json
      * VerifiedSearch/processed_graphs/synthetic/<graph>/graph.json
      * an absolute path

    We try the corresponding roots and return the first existing path.
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {resolved}")
        return resolved

    processed_root = manifest_path.parent.resolve()
    verified_search_root = processed_root.parent
    project_root = verified_search_root.parent

    candidates = [
        processed_root / path,       # synthetic/...
        verified_search_root / path, # processed_graphs/...
        project_root / path,         # VerifiedSearch/processed_graphs/...
        Path.cwd() / path,
    ]

    # Also recover from manifests containing a duplicated/overqualified prefix.
    parts = path.parts
    for dataset_anchor in ("synthetic", "real"):
        if dataset_anchor in parts:
            anchor_index = parts.index(dataset_anchor)
            candidates.append(processed_root / Path(*parts[anchor_index:]))

    seen: set[Path] = set()
    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        checked.append(resolved)
        if resolved.exists():
            return resolved

    checked_text = "\n  - ".join(str(item) for item in checked)
    raise FileNotFoundError(
        f"Could not resolve manifest path: {raw_path}\n"
        f"Checked:\n  - {checked_text}"
    )


def canonical_diffusion_name(diffusion: str, uniform_p: float) -> str:
    if diffusion == "uniform":
        return f"uniform_ic_p{uniform_p:g}"
    return "weighted_cascade"


def diffusion_description(diffusion: str, uniform_p: float) -> str:
    if diffusion == "uniform":
        return (
            "Independent Cascade (IC): every directed edge activates its target "
            f"with probability p={uniform_p:g}."
        )
    return (
        "Weighted Cascade IC: a directed edge u->v activates v with probability "
        "1 / in_degree(v)."
    )


def pairwise_jaccard(rankings: Sequence[Sequence[int]], k: int) -> float | None:
    sets = [set(ranking[:k]) for ranking in rankings if ranking]
    if len(sets) < 2:
        return None
    values: list[float] = []
    for left, right in itertools.combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return statistics.fmean(values) if values else None


def normalize_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    match = re.fullmatch(
        r"(?:node\s*)?[#:]?\s*([+-]?\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Prompting and output parsing
# ---------------------------------------------------------------------------


def build_candidate_prompt(
    adjacency_text: str,
    graph_name: str,
    num_nodes: int,
    num_edges: int,
    list_length: int,
    max_budget: int,
    diffusion: str,
    uniform_p: float,
) -> str:
    return f"""You are selecting candidate seed nodes for influence maximization.

Task
----
Given the complete directed graph below, rank {list_length} distinct candidate
nodes that are likely to form high-quality seed sets for budgets up to
{max_budget}. The ranking must be useful for several budgets, so earlier nodes
must have stronger expected marginal value and later nodes should add coverage
rather than repeat the same region.

Diffusion model
---------------
{diffusion_description(diffusion, uniform_p)}

Selection criteria
------------------
1. Outgoing and multi-hop reachability.
2. Coverage of different graph regions or communities.
3. Low redundancy with nodes ranked earlier.
4. Bridge positions that can reach otherwise separated regions.
5. Robust usefulness across seed budgets from 1 to {max_budget}.

Output contract
---------------
Return JSON only, with exactly one key named \"ranked_candidates\".\n/no_think
The value must be a list of {list_length} distinct integer node IDs.
Every node ID must be between 1 and {num_nodes}.
Do not include explanations, Markdown, comments, or additional keys.

Required format:
{{"ranked_candidates": [17, 4, 91]}}

Graph metadata
--------------
Name: {graph_name}
Nodes: {num_nodes}
Edges: {num_edges}

Graph adjacency list
--------------------
{adjacency_text}
"""


def strip_thinking_content(text: str) -> str:
    """Remove complete Qwen thinking blocks and safely handle truncated ones."""
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # If generation was truncated inside an open thinking block, there is no
    # final answer to parse. Removing it makes the failure explicit instead of
    # accidentally parsing numbers from the reasoning text.
    cleaned = re.sub(
        r"<think>.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip()


def strip_code_fences(text: str) -> str:
    text = strip_thinking_content(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _literal_eval_candidate(text: str) -> Any:
    """Accept Python-style lists/dicts with single quotes as a fallback."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def extract_first_json_value(text: str) -> Any:
    """Extract the first decodable JSON/Python object or recover a truncated list."""
    cleaned = strip_code_fences(text)
    decoder = json.JSONDecoder()

    # First try the complete response as strict JSON, then Python literal syntax.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        literal = _literal_eval_candidate(cleaned)
        if literal is not None:
            return literal

    # Then scan from each likely JSON start.
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        fragment = cleaned[index:]
        try:
            value, _ = decoder.raw_decode(fragment)
            return value
        except json.JSONDecodeError:
            literal = _literal_eval_candidate(fragment)
            if literal is not None:
                return literal

    # Recover the useful prefix of a malformed or truncated ranked_candidates
    # list. Only inspect text after the expected key to avoid parsing unrelated
    # numbers from explanations.
    marker = re.search(
        r"[\"']?ranked_candidates[\"']?\s*[:=]\s*\[",
        cleaned,
        flags=re.IGNORECASE,
    )
    if marker:
        tail = cleaned[marker.end():]
        tail = tail.split("]", 1)[0]
        recovered = [int(token) for token in re.findall(r"(?<![\w.])-?\d+", tail)]
        if recovered:
            return {"ranked_candidates": recovered}

    # Last-resort recovery for a bare bracketed list.
    bare = re.search(r"\[([^\]]+)\]", cleaned, flags=re.DOTALL)
    if bare:
        recovered = [int(token) for token in re.findall(r"(?<![\w.])-?\d+", bare.group(1))]
        if recovered:
            return recovered

    raise ValueError("No valid JSON object, list, or recoverable candidate list found.")


def parse_ranked_candidates(
    raw_output: str,
    num_nodes: int,
    desired_length: int,
) -> tuple[list[int], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "invalid_items": [],
        "out_of_range_items": [],
        "duplicate_items": [],
    }
    value = extract_first_json_value(raw_output)

    if isinstance(value, dict):
        if "ranked_candidates" not in value:
            raise ValueError("JSON object does not contain 'ranked_candidates'.")
        items = value["ranked_candidates"]
    elif isinstance(value, list):
        items = value
        diagnostics["accepted_bare_list"] = True
    else:
        raise ValueError("Parsed JSON must be an object or list.")

    if not isinstance(items, list):
        raise ValueError("'ranked_candidates' must be a list.")

    candidates: list[int] = []
    seen: set[int] = set()
    for item in items:
        if isinstance(item, dict):
            item = item.get("node", item.get("id"))
        node = normalize_int(item)
        if node is None:
            diagnostics["invalid_items"].append(item)
            continue
        if not 1 <= node <= num_nodes:
            diagnostics["out_of_range_items"].append(node)
            continue
        if node in seen:
            diagnostics["duplicate_items"].append(node)
            continue
        seen.add(node)
        candidates.append(node)
        if len(candidates) == desired_length:
            break

    diagnostics["parsed_count"] = len(candidates)
    diagnostics["requested_count"] = desired_length
    return candidates, diagnostics


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    text: str
    metadata: dict[str, Any]


class CandidateBackend:
    name = "base"

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        permutation: dict[str, Any],
        list_length: int,
    ) -> GenerationResult:
        raise NotImplementedError


class DegreeBackend(CandidateBackend):
    """Deterministic smoke-test backend; not an experimental LLM baseline."""

    name = "degree_smoke_test"

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        permutation: dict[str, Any],
        list_length: int,
    ) -> GenerationResult:
        del prompt, seed
        n = int(permutation["num_nodes"])
        out_degree = [0] * (n + 1)
        in_degree = [0] * (n + 1)
        for source, target in permutation["edges"]:
            out_degree[int(source)] += 1
            in_degree[int(target)] += 1
        ranking = sorted(
            range(1, n + 1),
            key=lambda node: (-out_degree[node], -in_degree[node], node),
        )[:list_length]
        return GenerationResult(
            text=json.dumps({"ranked_candidates": ranking}),
            metadata={
                "backend": self.name,
                "warning": "Smoke-test backend only; do not report as an LLM result.",
            },
        )


class HuggingFaceBackend(CandidateBackend):
    name = "huggingface"

    def __init__(
        self,
        model_name: str,
        torch_dtype: str,
        device_map: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        max_input_tokens: int | None,
        local_files_only: bool,
        trust_remote_code: bool,
        enable_thinking: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            try:
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError:
                AutoModelForMultimodalLM = None
                AutoProcessor = None
        except ImportError as exc:
            raise RuntimeError(
                "The hf backend requires torch and transformers. Install them first."
            ) from exc

        self.torch = torch
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens_override = max_input_tokens
        self.enable_thinking = enable_thinking

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map[torch_dtype]

        LOGGER.info("Inspecting model config: %s", model_name)
        config = AutoConfig.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        architectures = set(getattr(config, "architectures", []) or [])
        model_type = str(getattr(config, "model_type", ""))
        self.is_multimodal = (
            model_type == "qwen3_5"
            or "Qwen3_5ForConditionalGeneration" in architectures
        )

        common_kwargs = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }

        if self.is_multimodal:
            if AutoProcessor is None or AutoModelForMultimodalLM is None:
                raise RuntimeError(
                    "Qwen3.5 requires a recent Transformers build exposing "
                    "AutoProcessor and AutoModelForMultimodalLM. Install the "
                    "latest Transformers main branch as documented by Qwen."
                )
            LOGGER.info("Loading multimodal processor: %s", model_name)
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                **common_kwargs,
            )
            self.tokenizer = self.processor.tokenizer
            self.chat_handler = self.processor
            LOGGER.info("Loading multimodal model: %s", model_name)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                **common_kwargs,
            )
        else:
            self.processor = None
            LOGGER.info("Loading tokenizer: %s", model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                **common_kwargs,
            )
            self.chat_handler = self.tokenizer
            LOGGER.info("Loading causal language model: %s", model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                **common_kwargs,
            )
        self.model.eval()
        self.model_name = model_name

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _render_chat(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise graph-optimization assistant. Follow the "
                    "user's JSON output contract exactly. Do not reveal reasoning. "
                    "/no_think"
                ),
            },
            {"role": "user", "content": f"/no_think\n{prompt}"},
        ]
        if hasattr(self.chat_handler, "apply_chat_template"):
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            # Qwen3/Qwen3.5 support explicit thinking control. Other templates
            # simply ignore this only if accepted, so add it conditionally.
            if "qwen" in self.model_name.lower():
                template_kwargs["enable_thinking"] = self.enable_thinking
            return self.chat_handler.apply_chat_template(messages, **template_kwargs)
        return (
            "System: You are a precise graph-optimization assistant.\n"
            f"User: {prompt}\nAssistant:"
        )

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        permutation: dict[str, Any],
        list_length: int,
    ) -> GenerationResult:
        del permutation, list_length
        from transformers import set_seed

        set_seed(seed)
        rendered = self._render_chat(prompt)
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_tokens = int(encoded["input_ids"].shape[-1])
        text_config = getattr(self.model.config, "text_config", None)
        model_limit = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(text_config, "max_position_embeddings", None)
        )
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)

        limits = [
            limit
            for limit in (
                self.max_input_tokens_override,
                model_limit,
                tokenizer_limit if isinstance(tokenizer_limit, int) and tokenizer_limit < 10**9 else None,
            )
            if limit is not None
        ]
        effective_limit = min(limits) if limits else None
        if effective_limit is not None and input_tokens + self.max_new_tokens > effective_limit:
            raise ValueError(
                f"Prompt is too long: {input_tokens} input tokens + "
                f"{self.max_new_tokens} requested output tokens exceeds limit "
                f"{effective_limit}. The script does not silently truncate graphs."
            )

        try:
            input_device = next(self.model.parameters()).device
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
        except StopIteration:
            pass

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generation_kwargs)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated_ids = output_ids[0, input_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            metadata={
                "backend": self.name,
                "model_name": self.model_name,
                "input_tokens": input_tokens,
                "output_tokens": int(generated_ids.shape[-1]),
                "generation_seconds": elapsed,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "enable_thinking": self.enable_thinking,
                "is_multimodal_model": self.is_multimodal,
            },
        )


# ---------------------------------------------------------------------------
# Candidate generation and aggregation
# ---------------------------------------------------------------------------


def map_to_internal(
    permuted_candidates: Sequence[int],
    permutation: dict[str, Any],
) -> list[int]:
    inverse = permutation["permuted_to_internal"]
    return [int(inverse[node - 1]) for node in permuted_candidates]


def run_one_permutation(
    backend: CandidateBackend,
    graph: dict[str, Any],
    permutation_path: Path,
    diffusion: str,
    uniform_p: float,
    list_length: int,
    max_budget: int,
    base_seed: int,
    max_retries: int,
) -> dict[str, Any]:
    permutation = load_json(permutation_path)
    text_path = permutation_path.with_suffix(".txt")
    if not text_path.exists():
        raise FileNotFoundError(f"Missing permutation adjacency file: {text_path}")
    adjacency_text = text_path.read_text(encoding="utf-8")

    prompt = build_candidate_prompt(
        adjacency_text=adjacency_text,
        graph_name=str(graph["name"]),
        num_nodes=int(graph["num_nodes"]),
        num_edges=int(graph["num_edges"]),
        list_length=list_length,
        max_budget=max_budget,
        diffusion=diffusion,
        uniform_p=uniform_p,
    )

    permutation_index = int(permutation["permutation_index"])
    attempts: list[dict[str, Any]] = []
    parsed: list[int] = []
    final_diagnostics: dict[str, Any] = {}

    for retry in range(max_retries + 1):
        attempt_seed = base_seed + permutation_index * 100 + retry
        retry_prompt = prompt
        if retry > 0:
            retry_prompt += (
                "\nCORRECTION: Your previous response could not be parsed or did not "
                f"contain enough valid unique IDs. Return only the required JSON with "
                f"{list_length} distinct IDs."
            )

        started = time.perf_counter()
        generation = backend.generate(
            retry_prompt,
            seed=attempt_seed,
            permutation=permutation,
            list_length=list_length,
        )
        wall_seconds = time.perf_counter() - started

        attempt_record: dict[str, Any] = {
            "retry": retry,
            "seed": attempt_seed,
            "raw_output": generation.text,
            "backend_metadata": generation.metadata,
            "wall_seconds": wall_seconds,
        }
        try:
            parsed, diagnostics = parse_ranked_candidates(
                generation.text,
                num_nodes=int(graph["num_nodes"]),
                desired_length=list_length,
            )
            attempt_record["parse_diagnostics"] = diagnostics
            final_diagnostics = diagnostics
            attempts.append(attempt_record)
            if len(parsed) >= max_budget:
                break
        except Exception as exc:  # Keep raw output for auditability.
            attempt_record["parse_error"] = str(exc)
            attempts.append(attempt_record)

    internal_candidates = map_to_internal(parsed, permutation) if parsed else []
    status = "ok" if len(parsed) >= max_budget else "insufficient_candidates"

    return {
        "graph_name": graph["name"],
        "dataset_type": graph["dataset_type"],
        "diffusion": canonical_diffusion_name(diffusion, uniform_p),
        "permutation_index": permutation_index,
        "permutation_file": str(permutation_path),
        "status": status,
        "requested_list_length": list_length,
        "minimum_required": max_budget,
        "permuted_candidates": parsed,
        "internal_candidates": internal_candidates,
        "parse_diagnostics": final_diagnostics,
        "attempts": attempts,
    }


def aggregate_rankings(
    graph: dict[str, Any],
    runs: Sequence[dict[str, Any]],
    budgets: Sequence[int],
    pool_multiplier: int,
) -> dict[str, Any]:
    successful_rankings = [
        list(map(int, run["internal_candidates"]))
        for run in runs
        if run["status"] == "ok" and run["internal_candidates"]
    ]
    partial_rankings = [
        list(map(int, run["internal_candidates"]))
        for run in runs
        if run["internal_candidates"]
    ]
    rankings = successful_rankings or partial_rankings

    if not rankings:
        raise RuntimeError("No permutation produced any valid candidates.")

    scores: dict[int, float] = {}
    appearances: dict[int, int] = {}
    ranks_by_node: dict[int, list[int]] = {}

    for ranking in rankings:
        for rank, node in enumerate(ranking, start=1):
            scores[node] = scores.get(node, 0.0) + 1.0 / math.log2(rank + 1.0)
            appearances[node] = appearances.get(node, 0) + 1
            ranks_by_node.setdefault(node, []).append(rank)

    denominator = len(rankings)
    original_by_internal = {
        int(item["internal"]): item["original"] for item in graph["id_map"]
    }

    ranked_nodes = sorted(
        scores,
        key=lambda node: (
            -(scores[node] / denominator),
            -appearances[node],
            statistics.fmean(ranks_by_node[node]),
            node,
        ),
    )

    max_budget = max(budgets)
    pool_size = min(int(graph["num_nodes"]), pool_multiplier * max_budget)
    candidate_pool_ids = ranked_nodes[:pool_size]

    candidate_pool = []
    for node in candidate_pool_ids:
        candidate_pool.append(
            {
                "internal_node": node,
                "original_node": original_by_internal.get(node),
                "consensus_score": scores[node] / denominator,
                "appearance_count": appearances[node],
                "appearance_rate": appearances[node] / denominator,
                "mean_rank_when_present": statistics.fmean(ranks_by_node[node]),
                "best_rank": min(ranks_by_node[node]),
                "worst_rank": max(ranks_by_node[node]),
            }
        )

    stability = {
        f"top_{budget}_mean_pairwise_jaccard": pairwise_jaccard(rankings, budget)
        for budget in budgets
    }

    consensus_seeds = {
        str(budget): candidate_pool_ids[:budget]
        for budget in budgets
        if budget <= len(candidate_pool_ids)
    }

    return {
        "num_runs_total": len(runs),
        "num_rankings_used": len(rankings),
        "num_fully_valid_runs": len(successful_rankings),
        "candidate_pool_size": len(candidate_pool_ids),
        "candidate_pool_ids": candidate_pool_ids,
        "candidate_pool": candidate_pool,
        "consensus_seed_baselines": consensus_seeds,
        "permutation_stability": stability,
        "aggregation": {
            "score": "mean over permutations of 1/log2(rank+1)",
            "tie_break": [
                "higher appearance count",
                "lower mean rank",
                "lower canonical internal node ID",
            ],
        },
    }


def process_graph(
    entry: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    backend: CandidateBackend,
    diffusion: str,
    uniform_p: float,
    budgets: Sequence[int],
    list_multiplier: int,
    pool_multiplier: int,
    base_seed: int,
    max_retries: int,
    overwrite: bool,
) -> dict[str, Any]:
    graph_path = resolve_manifest_path(entry["processed_file"], manifest_path)
    graph = load_json(graph_path)

    max_budget = max(budgets)
    if max_budget > int(graph["num_nodes"]):
        raise ValueError(
            f"Maximum budget {max_budget} exceeds N={graph['num_nodes']} for "
            f"graph {graph['name']}."
        )
    list_length = min(int(graph["num_nodes"]), list_multiplier * max_budget)

    diffusion_name = canonical_diffusion_name(diffusion, uniform_p)
    graph_output_dir = (
        output_dir
        / safe_slug(str(getattr(backend, "model_name", backend.name)))
        / safe_slug(str(graph["dataset_type"]))
        / safe_slug(str(graph["name"]))
        / diffusion_name
    )
    aggregate_path = graph_output_dir / "aggregate.json"
    if aggregate_path.exists() and not overwrite:
        LOGGER.info("Skipping existing result: %s", aggregate_path)
        return load_json(aggregate_path)

    permutation_paths = [
        resolve_manifest_path(path, manifest_path)
        for path in entry.get("permutation_files", [])
    ]
    if not permutation_paths:
        raise ValueError(
            f"No permutation files listed for {graph['name']}. Re-run preprocessing "
            "with --num-permutations >= 1."
        )

    runs: list[dict[str, Any]] = []
    for permutation_path in permutation_paths:
        LOGGER.info(
            "Graph=%s diffusion=%s permutation=%s",
            graph["name"],
            diffusion_name,
            permutation_path.stem,
        )
        run = run_one_permutation(
            backend=backend,
            graph=graph,
            permutation_path=permutation_path,
            diffusion=diffusion,
            uniform_p=uniform_p,
            list_length=list_length,
            max_budget=max_budget,
            base_seed=base_seed,
            max_retries=max_retries,
        )
        runs.append(run)
        save_json(run, graph_output_dir / "runs" / f"{permutation_path.stem}.json")

    aggregate = aggregate_rankings(
        graph=graph,
        runs=runs,
        budgets=budgets,
        pool_multiplier=pool_multiplier,
    )
    aggregate.update(
        {
            "graph_name": graph["name"],
            "dataset_type": graph["dataset_type"],
            "graph_file": str(graph_path),
            "num_nodes": graph["num_nodes"],
            "num_edges": graph["num_edges"],
            "diffusion": diffusion_name,
            "budgets": list(budgets),
            "ranked_list_length": list_length,
            "backend": backend.name,
            "model_name": getattr(backend, "model_name", None),
            "run_files": [
                str(graph_output_dir / "runs" / f"perm_{i:03d}.json")
                for i in range(len(runs))
            ],
        }
    )
    save_json(aggregate, aggregate_path)
    return aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate permutation-symmetrized candidate rankings for the "
            "verified-search influence-maximization pipeline."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--backend",
        choices=("hf", "degree"),
        default="degree",
        help=(
            "Use 'degree' first to smoke-test paths; use 'hf' for the actual LLM "
            "experiment."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.5-9B",
        help="Hugging Face model ID or local model directory for --backend hf.",
    )
    parser.add_argument(
        "--diffusion",
        choices=("uniform", "weighted_cascade", "both"),
        default="both",
    )
    parser.add_argument("--uniform-p", type=float, default=0.1)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument(
        "--list-multiplier",
        type=int,
        default=2,
        help="Ranked-list length = multiplier * maximum budget.",
    )
    parser.add_argument(
        "--pool-multiplier",
        type=int,
        default=5,
        help="Candidate-pool size = multiplier * maximum budget.",
    )
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--graphs",
        nargs="*",
        default=None,
        help="Optional graph names to process. Omit to process all graphs.",
    )
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        choices=("synthetic", "real"),
        default=["synthetic", "real"],
    )
    parser.add_argument("--overwrite", action="store_true")

    # Hugging Face settings.
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Default 0.0 uses deterministic decoding so variation primarily comes "
            "from graph permutations."
        ),
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable model reasoning blocks when supported. Keep disabled for the "
            "main permutation-robustness experiment so outputs remain concise JSON."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.budgets or any(budget <= 0 for budget in args.budgets):
        raise ValueError("All budgets must be positive integers.")
    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Budgets must not contain duplicates.")
    if args.list_multiplier < 1 or args.pool_multiplier < 1:
        raise ValueError("List and pool multipliers must be at least 1.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be nonnegative.")
    if not 0 < args.uniform_p <= 1:
        raise ValueError("--uniform-p must be in (0, 1].")
    if args.temperature < 0:
        raise ValueError("--temperature must be nonnegative.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")


def make_backend(args: argparse.Namespace) -> CandidateBackend:
    if args.backend == "degree":
        return DegreeBackend()
    return HuggingFaceBackend(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        enable_thinking=args.enable_thinking,
    )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        validate_args(args)
        manifest_path = args.manifest.resolve()
        manifest = load_json(manifest_path)
        backend = make_backend(args)

        selected_names = set(args.graphs) if args.graphs else None
        entries = [
            entry
            for entry in manifest.get("graphs", [])
            if entry.get("dataset_type") in set(args.dataset_types)
            and (selected_names is None or entry.get("name") in selected_names)
        ]
        if not entries:
            raise ValueError("No manifest entries match the selected filters.")

        diffusions = (
            ["uniform", "weighted_cascade"]
            if args.diffusion == "both"
            else [args.diffusion]
        )

        summaries: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        args.output_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            for diffusion in diffusions:
                try:
                    aggregate = process_graph(
                        entry=entry,
                        manifest_path=manifest_path,
                        output_dir=args.output_dir.resolve(),
                        backend=backend,
                        diffusion=diffusion,
                        uniform_p=args.uniform_p,
                        budgets=sorted(args.budgets),
                        list_multiplier=args.list_multiplier,
                        pool_multiplier=args.pool_multiplier,
                        base_seed=args.base_seed,
                        max_retries=args.max_retries,
                        overwrite=args.overwrite,
                    )
                    summaries.append(
                        {
                            "graph_name": aggregate["graph_name"],
                            "dataset_type": aggregate["dataset_type"],
                            "diffusion": aggregate["diffusion"],
                            "backend": aggregate["backend"],
                            "candidate_pool_size": aggregate["candidate_pool_size"],
                            "num_fully_valid_runs": aggregate["num_fully_valid_runs"],
                            "permutation_stability": aggregate["permutation_stability"],
                        }
                    )
                    LOGGER.info(
                        "Finished graph=%s diffusion=%s pool=%s valid_runs=%s",
                        aggregate["graph_name"],
                        aggregate["diffusion"],
                        aggregate["candidate_pool_size"],
                        aggregate["num_fully_valid_runs"],
                    )
                except Exception as exc:
                    LOGGER.exception(
                        "Failed graph=%s diffusion=%s", entry.get("name"), diffusion
                    )
                    failures.append(
                        {
                            "graph_name": str(entry.get("name")),
                            "diffusion": diffusion,
                            "error": str(exc),
                        }
                    )

        index = {
            "manifest": str(manifest_path),
            "backend": backend.name,
            "model_name": args.model_name if args.backend == "hf" else None,
            "budgets": sorted(args.budgets),
            "uniform_p": args.uniform_p,
            "summaries": summaries,
            "failures": failures,
        }
        save_json(index, args.output_dir.resolve() / "index.json")

        print(
            f"Completed {len(summaries)} graph/diffusion job(s); "
            f"{len(failures)} failure(s)."
        )
        print(f"Output index: {args.output_dir.resolve() / 'index.json'}")
        return 0 if summaries else 1
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
