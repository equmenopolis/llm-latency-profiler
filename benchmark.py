#!/usr/bin/env python3
"""Benchmark OpenAI and Anthropic latency across models, tiers, and prompt sizes.

The benchmark measures:
  * TTFT: request start to the first non-empty output-text delta.
  * Full latency: request start to the completed response.

Requests run sequentially so client-side concurrency does not distort latency.
Within each repetition, model/tier/scenario combinations are shuffled to reduce
ordering and time-of-day bias. Each repetition uses randomly selected,
paragraph-aligned excerpts from a long-form corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
import tiktoken
from openai import OpenAI


# Active general text/reasoning models in the OpenAI model catalog as of
# 2026-10-26. Specialized image, audio, embedding, moderation, cybersecurity,
# life-sciences, and open-weight models are intentionally omitted because they
# are not comparable through this general text-streaming Responses test.
OPENAI_CURRENT_TEXT_MODELS = (
    "gpt-6-astra",
    "gpt-6.1-sol",
    "gpt-6.1-terra",
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.2",
    "gpt-5.2-pro",
)
# Deprecated but not yet retired general text/reasoning models. Keeping these in
# --all-models makes the benchmark useful for migration comparisons while the
# provider still accepts requests to them.
OPENAI_DEPRECATED_TEXT_MODELS = (
    "gpt-5.4-nano",
    "gpt-5.3-codex",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "o3-pro",
    "o3",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
)
OPENAI_TEXT_MODELS = OPENAI_CURRENT_TEXT_MODELS + OPENAI_DEPRECATED_TEXT_MODELS

# Current generally available Claude lineup from Anthropic's models overview as
# of 2026-10-24.
ANTHROPIC_CURRENT_TEXT_MODELS = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
)
# Deprecated but still functional Claude models from Anthropic's lifecycle
# table. Access-controlled models can still fail for accounts without access.
ANTHROPIC_DEPRECATED_TEXT_MODELS = (
    "claude-mythos-preview",
    "claude-sonnet-4-5-20250929",
)
ANTHROPIC_TEXT_MODELS = (
    ANTHROPIC_CURRENT_TEXT_MODELS + ANTHROPIC_DEPRECATED_TEXT_MODELS
)
ALL_TEXT_MODELS = OPENAI_TEXT_MODELS + ANTHROPIC_TEXT_MODELS
MODEL_CATALOG_AS_OF = "2026-10-26"
# Based on OpenAI's Standard, Fast, and Ultrafast compatibility documentation
# and previously verified pairs as of 2026-10-26. Newly cataloged deprecated
# models are Standard-only unless current documentation confirms another tier.
# Keep this explicit: unsupported combinations must be removed before job
# creation rather than discovered by making billable calls. Unknown model IDs
# supplied with --models are also treated conservatively as Standard-only.
MODEL_TIER_SUPPORT: dict[str, tuple[str, ...]] = {
    "gpt-6-astra": ("normal", "fast", "ultrafast"),
    "gpt-6.1-sol": ("normal", "fast", "ultrafast"),
    "gpt-6.1-terra": ("normal", "fast"),
    "gpt-6-luna": ("normal", "fast"),
    "gpt-6-sol": ("normal", "fast"),
    "gpt-5.6-sol": ("normal", "fast"),
    "gpt-5.6-terra": ("normal", "fast"),
    "gpt-5.6-luna": ("normal", "fast"),
    "gpt-5.5": ("normal", "fast"),
    "gpt-5.5-pro": ("normal",),
    "gpt-5.4": ("normal", "fast"),
    "gpt-5.4-pro": ("normal",),
    "gpt-5.4-mini": ("normal", "fast"),
    "gpt-5.4-nano": ("normal",),
    "gpt-5.3-codex": ("normal", "fast"),
    "gpt-5.2": ("normal", "fast"),
    "gpt-5.2-pro": ("normal",),
    "gpt-5.1": ("normal",),
    "gpt-5": ("normal", "fast"),
    "gpt-5-mini": ("normal", "fast"),
    "gpt-5-nano": ("normal",),
    "gpt-5-pro": ("normal",),
    "o3-pro": ("normal",),
    "o3": ("normal", "fast"),
    "gpt-4.1": ("normal",),
    "gpt-4.1-mini": ("normal",),
    "gpt-4o": ("normal",),
    "gpt-4o-mini": ("normal",),
    # Anthropic Priority Tier is a contracted capacity tier rather than an
    # OpenAI Fast/Ultrafast analogue. Force Standard capacity for comparable
    # normal-mode measurements and never probe the other modes.
    "claude-fable-5-1": ("normal",),
    "claude-opus-5-5": ("normal",),
    "claude-sonnet-5-5": ("normal",),
    "claude-haiku-5-5": ("normal",),
    "claude-mythos-preview": ("normal",),
    "claude-sonnet-4-5-20250929": ("normal",),
}
MODEL_TIER_SUPPORT_AS_OF = "2026-10-26"
TIER_LABELS = {
    "normal": "default",
    "standard": "default",
    "default": "default",
    "fast": "fast",
    "priority": "fast",
    "ultrafast": "ultrafast",
}
CONTEXT_TARGETS = {"short": 200, "medium": 1_000, "long": 5_000}
DEFAULT_CORPUS_URL = "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
DEFAULT_CORPUS_TITLE = "Moby-Dick; or, The Whale by Herman Melville"
REPLY_CONFIG = {
    "short": {
        "instruction": (
            "Reply with exactly one sentence of 18 to 25 words that summarizes "
            "the main idea or event in the excerpt."
        ),
        # This budget includes hidden reasoning tokens as well as visible text.
        # Keep it comfortably above the requested sentence length so reasoning
        # models do not terminate with response.incomplete before emitting text.
        "max_output_tokens": 1_024,
    },
    "medium": {
        "instruction": (
            "Reply with exactly one paragraph of 100 to 130 words that summarizes "
            "the excerpt's main events, ideas, or themes. Do not use a heading, "
            "bullets, or multiple paragraphs."
        ),
        "max_output_tokens": 1_536,
    },
}
SCENARIO_NAMES = tuple(
    f"{context_size}_context_{reply_size}_reply"
    for context_size in CONTEXT_TARGETS
    for reply_size in REPLY_CONFIG
)

@dataclass(frozen=True)
class Scenario:
    scenario: str
    context_size: str
    context_tokens_target: int
    reply_size: str
    max_output_tokens: int
    response_instruction: str


@dataclass(frozen=True)
class Corpus:
    title: str
    source: str
    sha256: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class ContextSample:
    text: str
    tokens_actual: int
    start_paragraph: int
    end_paragraph: int
    sha256: str


@dataclass(frozen=True)
class Job:
    run: int
    provider: str
    model: str
    tier_label: str
    service_tier: str
    scenario: Scenario
    prompt: str
    prompt_tokens_estimate: int
    context: ContextSample


RAW_FIELDS = (
    "timestamp_utc",
    "run",
    "attempt",
    "provider",
    "model",
    "tier",
    "requested_service_tier",
    "actual_service_tier",
    "tier_valid",
    "scenario",
    "context_size",
    "context_tokens_target",
    "context_tokens_actual",
    "context_start_paragraph",
    "context_end_paragraph",
    "context_sha256",
    "prompt_tokens_estimate",
    "reply_size",
    "max_output_tokens",
    "status",
    "response_status",
    "incomplete_reason",
    "ttft_seconds",
    "full_seconds",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "response_id",
    "output_characters",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OpenAI and Anthropic model latency across supported normal, "
            "fast, and ultrafast modes."
        )
    )
    model_selection = parser.add_mutually_exclusive_group(required=True)
    model_selection.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Benchmark only these model IDs.",
    )
    model_selection.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark every current OpenAI and Anthropic model in the built-in catalog.",
    )
    model_selection.add_argument(
        "--all-openai-models",
        action="store_true",
        help="Benchmark every current OpenAI model in the built-in catalog.",
    )
    model_selection.add_argument(
        "--all-anthropic-models",
        action="store_true",
        help="Benchmark every current Anthropic model in the built-in catalog.",
    )
    model_selection.add_argument(
        "--list-models",
        action="store_true",
        help="Print the built-in all-models catalog and exit.",
    )
    model_selection.add_argument(
        "--list-model-tiers",
        action="store_true",
        help="Print the built-in model/tier support matrix and exit.",
    )
    model_selection.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print the selectable benchmark scenarios and exit.",
    )
    model_selection.add_argument(
        "--show-prompt",
        choices=SCENARIO_NAMES,
        metavar="SCENARIO",
        help="Generate and print the run-1 prompt for one scenario, using --seed.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["normal", "fast", "ultrafast"],
        choices=sorted(TIER_LABELS),
        help="Processing modes to benchmark (default: normal fast ultrafast).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_NAMES,
        default=list(SCENARIO_NAMES),
        help="Scenarios to run (default: all six).",
    )
    parser.add_argument(
        "--runs",
        "--runs-per-test",
        dest="runs",
        type=positive_int,
        default=10,
        help="Generations targeted for each model/tier/scenario test (default: 10).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help=(
            "Optional OpenAI reasoning effort. Anthropic models use their API "
            "default (default: unset)."
        ),
    )
    corpus_source = parser.add_mutually_exclusive_group()
    corpus_source.add_argument(
        "--corpus-file",
        type=Path,
        help="Use a local UTF-8 text corpus instead of the default Project Gutenberg book.",
    )
    corpus_source.add_argument(
        "--corpus-url",
        help=f"Download a text corpus from this URL (default: {DEFAULT_CORPUS_URL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result directory (default: benchmark-results/<UTC timestamp>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261023,
        help="Seed used to shuffle each round (default: %(default)s).",
    )
    parser.add_argument(
        "--pause",
        type=nonnegative_float,
        default=0.25,
        help="Seconds to pause between API calls (default: 0.25).",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=120.0,
        help=(
            "Hard wall-clock timeout for each generation attempt, including the "
            "entire stream (default: 120)."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=3,
        help="Maximum API attempts for an errored or downgraded job (default: 3).",
    )
    parser.add_argument(
        "--allow-cache",
        action="store_true",
        help="Reuse identical prompts. Default behavior adds an early nonce to avoid prompt-cache hits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and inspect the matrix without making API requests.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def selected_models(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        models = list(ALL_TEXT_MODELS)
    elif args.all_openai_models:
        models = list(OPENAI_TEXT_MODELS)
    elif args.all_anthropic_models:
        models = list(ANTHROPIC_TEXT_MODELS)
    else:
        models = list(args.models or [])
    # Preserve the user's order while preventing accidental duplicate spend.
    return list(dict.fromkeys(models))


def provider_for_model(model: str) -> str:
    """Resolve built-in and custom model IDs to an API provider."""
    if model in ANTHROPIC_TEXT_MODELS or model.startswith("claude-"):
        return "anthropic"
    return "openai"


def provider_service_tier(provider: str, service_tier: str) -> str:
    if provider == "anthropic":
        return "standard_only"
    return service_tier


def encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg's license header and footer when present."""
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    body_start = start.end() if start else 0
    body_end = end.start() if end and end.start() > body_start else len(text)
    return text[body_start:body_end].strip()


def load_corpus(corpus_file: Path | None, corpus_url: str | None) -> Corpus:
    if corpus_file:
        raw_text = corpus_file.read_text(encoding="utf-8")
        source = str(corpus_file.resolve())
        title = corpus_file.stem
    else:
        source = corpus_url or DEFAULT_CORPUS_URL
        try:
            response = httpx.get(
                source,
                headers={"User-Agent": "OpenAI-latency-benchmark/1.0"},
                follow_redirects=True,
                timeout=60,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OSError(f"Could not download corpus from {source}: {exc}") from exc
        raw_text = response.content.decode("utf-8-sig")
        title = DEFAULT_CORPUS_TITLE if source == DEFAULT_CORPUS_URL else source

    body = strip_gutenberg_boilerplate(raw_text.replace("\r\n", "\n"))
    if source == DEFAULT_CORPUS_URL:
        # This edition repeats the chapter list before the novel. Start at the
        # second Chapter 1 heading so samples come from the narrative itself,
        # not the table of contents or Gutenberg front matter.
        chapter_one = list(re.finditer(r"CHAPTER 1\. Loomings\.", body, re.IGNORECASE))
        if len(chapter_one) >= 2:
            body = body[chapter_one[1].start() :]
    paragraphs = tuple(
        re.sub(r"\s*\n\s*", " ", block).strip()
        for block in re.split(r"\n\s*\n+", body)
        if block.strip()
    )
    if len(paragraphs) < 20:
        raise ValueError("Corpus must contain at least 20 blank-line-separated paragraphs.")
    return Corpus(
        title=title,
        source=source,
        sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        paragraphs=paragraphs,
    )


def sample_context(
    corpus: Corpus,
    target_tokens: int,
    enc: tiktoken.Encoding,
    rng: random.Random,
    excluded_starts: set[int] | None = None,
) -> ContextSample:
    """Choose a random paragraph-aligned excerpt close to the token target."""
    all_candidate_starts = [
        index
        for index, paragraph in enumerate(corpus.paragraphs)
        if len(paragraph.split()) >= 20 and paragraph.upper() != paragraph
    ]
    candidate_starts = [
        index for index in all_candidate_starts if index not in (excluded_starts or set())
    ]
    if not candidate_starts:
        candidate_starts = all_candidate_starts
    rng.shuffle(candidate_starts)

    for start in candidate_starts:
        first_paragraph_tokens = len(enc.encode(corpus.paragraphs[start]))
        if first_paragraph_tokens > int(target_tokens * 1.25):
            continue
        pieces: list[str] = []
        previous: tuple[str, int, int] | None = None
        for end in range(start, len(corpus.paragraphs)):
            pieces.append(corpus.paragraphs[end])
            text = "\n\n".join(pieces)
            token_count = len(enc.encode(text))
            if token_count >= target_tokens:
                chosen_text = text
                chosen_tokens = token_count
                chosen_end = end
                if previous is not None:
                    previous_text, previous_tokens, previous_end = previous
                    previous_is_close = previous_tokens >= int(target_tokens * 0.85)
                    if previous_is_close and abs(previous_tokens - target_tokens) < abs(
                        token_count - target_tokens
                    ):
                        chosen_text = previous_text
                        chosen_tokens = previous_tokens
                        chosen_end = previous_end
                return ContextSample(
                    text=chosen_text,
                    tokens_actual=chosen_tokens,
                    start_paragraph=start,
                    end_paragraph=chosen_end,
                    sha256=hashlib.sha256(chosen_text.encode("utf-8")).hexdigest(),
                )
            previous = (text, token_count, end)

    raise ValueError(f"Corpus is too short to sample approximately {target_tokens} tokens.")


def render_prompt(context: str, response_instruction: str) -> str:
    return (
        "Read the reference excerpt, then follow the response instruction.\n\n"
        f"REFERENCE EXCERPT\n{context}\n\n"
        f"RESPONSE INSTRUCTION\n{response_instruction}"
    )


def build_scenarios(selected: Iterable[str] | None = None) -> list[Scenario]:
    selected_names = set(selected or SCENARIO_NAMES)
    scenarios: list[Scenario] = []
    for context_size, target_tokens in CONTEXT_TARGETS.items():
        for reply_size, reply in REPLY_CONFIG.items():
            scenario_name = f"{context_size}_context_{reply_size}_reply"
            if scenario_name not in selected_names:
                continue
            scenarios.append(
                Scenario(
                    scenario=scenario_name,
                    context_size=context_size,
                    context_tokens_target=target_tokens,
                    reply_size=reply_size,
                    max_output_tokens=int(reply["max_output_tokens"]),
                    response_instruction=str(reply["instruction"]),
                )
            )
    return scenarios


def normalize_tiers(values: Iterable[str]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        service_tier = TIER_LABELS[value]
        label = "normal" if service_tier == "default" else service_tier
        if service_tier not in seen:
            normalized.append((label, service_tier))
            seen.add(service_tier)
    return normalized


def supported_tier_labels(model: str) -> tuple[str, ...]:
    """Return known supported tier labels; custom model IDs default to Standard."""
    return MODEL_TIER_SUPPORT.get(model, ("normal",))


def unsupported_model_tiers(
    models: Iterable[str], tiers: Iterable[tuple[str, str]]
) -> list[tuple[str, str]]:
    return [
        (model, tier_label)
        for model in models
        for tier_label, _service_tier in tiers
        if tier_label not in supported_tier_labels(model)
    ]


def context_sample_manifest(jobs: Iterable[Job]) -> list[dict[str, Any]]:
    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for job in jobs:
        key = (job.run, job.scenario.context_size)
        unique[key] = {
            "run": job.run,
            "context_size": job.scenario.context_size,
            "target_tokens": job.scenario.context_tokens_target,
            "actual_tokens": job.context.tokens_actual,
            "start_paragraph": job.context.start_paragraph,
            "end_paragraph": job.context.end_paragraph,
            "sha256": job.context.sha256,
        }
    return [unique[key] for key in sorted(unique)]


def build_jobs(
    models: list[str],
    tiers: list[tuple[str, str]],
    scenarios: list[Scenario],
    runs: int,
    seed: int,
    corpus: Corpus,
    enc: tiktoken.Encoding,
) -> list[Job]:
    jobs: list[Job] = []
    used_starts: dict[str, set[int]] = defaultdict(set)
    for run in range(1, runs + 1):
        samples: dict[str, ContextSample] = {}
        for scenario in scenarios:
            if scenario.context_size not in samples:
                sample_rng = random.Random(
                    f"{seed}:{run}:{scenario.context_size}:{corpus.sha256}"
                )
                samples[scenario.context_size] = sample_context(
                    corpus,
                    scenario.context_tokens_target,
                    enc,
                    sample_rng,
                    used_starts[scenario.context_size],
                )
                used_starts[scenario.context_size].add(
                    samples[scenario.context_size].start_paragraph
                )

        rendered_scenarios = []
        for scenario in scenarios:
            sample = samples[scenario.context_size]
            prompt = render_prompt(sample.text, scenario.response_instruction)
            rendered_scenarios.append((scenario, prompt, len(enc.encode(prompt)), sample))

        round_jobs = [
            Job(
                run,
                provider_for_model(model),
                model,
                tier_label,
                provider_service_tier(provider_for_model(model), service_tier),
                scenario,
                prompt,
                prompt_tokens_estimate,
                sample,
            )
            for model in models
            for tier_label, service_tier in tiers
            if tier_label in supported_tier_labels(model)
            for scenario, prompt, prompt_tokens_estimate, sample in rendered_scenarios
        ]
        random.Random(seed + run).shuffle(round_jobs)
        jobs.extend(round_jobs)
    return jobs


def nested_attr(value: Any, *names: str, default: Any = None) -> Any:
    current = value
    for name in names:
        if current is None:
            return default
        current = current.get(name) if isinstance(current, dict) else getattr(current, name, None)
    return default if current is None else current


def tier_is_valid(provider: str, requested: str, actual: str | None) -> bool:
    if provider == "anthropic":
        return requested == "standard_only" and actual == "standard"
    if requested == "fast":
        return actual in {"fast", "priority"}
    return actual == requested


def make_input(job: Job, allow_cache: bool) -> str:
    if allow_cache:
        return job.prompt
    # An early unique value prevents the long common prefix from qualifying for a
    # prompt-cache hit, while leaving the semantic task unchanged.
    return f"Benchmark sample ID: {uuid.uuid4()}\n\n{job.prompt}"


class GenerationTimeoutError(TimeoutError):
    """Raised when one complete streamed attempt exceeds its wall-clock limit."""


@contextmanager
def generation_deadline(seconds: float) -> Iterable[None]:
    """Enforce a total attempt deadline, not only a socket read timeout.

    SIGALRM can interrupt a stream even when it continues receiving keepalive or
    non-text events. On platforms without interval timers, the SDK/httpx timeout
    remains the fallback.
    """
    can_alarm = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_alarm:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def handle_timeout(_signum: int, _frame: Any) -> None:
        raise GenerationTimeoutError(
            f"generation exceeded the {seconds:g}-second wall-clock timeout"
        )

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def call_openai_once(
    client: OpenAI,
    job: Job,
    attempt: int,
    reasoning_effort: str | None,
    allow_cache: bool,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": job.model,
        "input": make_input(job, allow_cache),
        "service_tier": job.service_tier,
        "max_output_tokens": job.scenario.max_output_tokens,
    }
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}

    started = time.perf_counter()
    first_text_at: float | None = None
    text_parts: list[str] = []
    terminal_response: Any | None = None

    with client.responses.stream(**request) as stream:
        for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    text_parts.append(delta)
            elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                terminal_response = getattr(event, "response", None)
        response = terminal_response or stream.get_final_response()
    finished = time.perf_counter()

    actual_tier = getattr(response, "service_tier", None)
    tier_valid = tier_is_valid(job.provider, job.service_tier, actual_tier)
    response_status = getattr(response, "status", None) or "completed"
    incomplete_reason = nested_attr(response, "incomplete_details", "reason", default="")
    if not tier_valid:
        status = "tier_mismatch"
        error = f"requested {job.service_tier}, received {actual_tier}"
    elif response_status == "incomplete":
        status = "incomplete"
        error = f"response incomplete: {incomplete_reason or 'reason not provided'}"
    elif response_status == "failed":
        status = "response_failed"
        error = "response failed"
    else:
        status = "ok"
        error = ""
    usage = getattr(response, "usage", None)
    output_text = "".join(text_parts) or getattr(response, "output_text", "") or ""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run": job.run,
        "attempt": attempt,
        "provider": job.provider,
        "model": job.model,
        "tier": job.tier_label,
        "requested_service_tier": job.service_tier,
        "actual_service_tier": actual_tier or "",
        "tier_valid": tier_valid,
        "scenario": job.scenario.scenario,
        "context_size": job.scenario.context_size,
        "context_tokens_target": job.scenario.context_tokens_target,
        "context_tokens_actual": job.context.tokens_actual,
        "context_start_paragraph": job.context.start_paragraph,
        "context_end_paragraph": job.context.end_paragraph,
        "context_sha256": job.context.sha256,
        "prompt_tokens_estimate": job.prompt_tokens_estimate,
        "reply_size": job.scenario.reply_size,
        "max_output_tokens": job.scenario.max_output_tokens,
        "status": status,
        "response_status": response_status,
        "incomplete_reason": incomplete_reason,
        "ttft_seconds": (first_text_at - started) if first_text_at is not None else "",
        "full_seconds": finished - started,
        "input_tokens": nested_attr(usage, "input_tokens", default=""),
        "cached_input_tokens": nested_attr(
            usage, "input_tokens_details", "cached_tokens", default=""
        ),
        "output_tokens": nested_attr(usage, "output_tokens", default=""),
        "response_id": getattr(response, "id", "") or "",
        "output_characters": len(output_text),
        "error": error,
    }


def call_anthropic_once(
    client: Any,
    job: Job,
    attempt: int,
    allow_cache: bool,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": job.model,
        "messages": [{"role": "user", "content": make_input(job, allow_cache)}],
        "service_tier": job.service_tier,
        "max_tokens": job.scenario.max_output_tokens,
    }

    started = time.perf_counter()
    first_text_at: float | None = None
    text_parts: list[str] = []
    with client.messages.stream(**request) as stream:
        for text_delta in stream.text_stream:
            if text_delta:
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                text_parts.append(text_delta)
        response = stream.get_final_message()
    finished = time.perf_counter()

    usage = getattr(response, "usage", None)
    # standard_only guarantees Standard capacity; current SDKs also expose the
    # assigned value as usage.service_tier.
    actual_tier = nested_attr(usage, "service_tier", default="standard")
    tier_valid = tier_is_valid(job.provider, job.service_tier, actual_tier)
    stop_reason = getattr(response, "stop_reason", None) or "end_turn"
    incomplete_reasons = {"max_tokens", "model_context_window_exceeded"}
    if not tier_valid:
        status = "tier_mismatch"
        error = f"requested {job.service_tier}, received {actual_tier}"
    elif stop_reason in incomplete_reasons:
        status = "incomplete"
        error = f"response incomplete: {stop_reason}"
    else:
        status = "ok"
        error = ""

    output_text = "".join(text_parts)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run": job.run,
        "attempt": attempt,
        "provider": job.provider,
        "model": job.model,
        "tier": job.tier_label,
        "requested_service_tier": job.service_tier,
        "actual_service_tier": actual_tier,
        "tier_valid": tier_valid,
        "scenario": job.scenario.scenario,
        "context_size": job.scenario.context_size,
        "context_tokens_target": job.scenario.context_tokens_target,
        "context_tokens_actual": job.context.tokens_actual,
        "context_start_paragraph": job.context.start_paragraph,
        "context_end_paragraph": job.context.end_paragraph,
        "context_sha256": job.context.sha256,
        "prompt_tokens_estimate": job.prompt_tokens_estimate,
        "reply_size": job.scenario.reply_size,
        "max_output_tokens": job.scenario.max_output_tokens,
        "status": status,
        "response_status": stop_reason,
        "incomplete_reason": stop_reason if stop_reason in incomplete_reasons else "",
        "ttft_seconds": (first_text_at - started) if first_text_at is not None else "",
        "full_seconds": finished - started,
        "input_tokens": nested_attr(usage, "input_tokens", default=""),
        "cached_input_tokens": nested_attr(
            usage, "cache_read_input_tokens", default=""
        ),
        "output_tokens": nested_attr(usage, "output_tokens", default=""),
        "response_id": getattr(response, "id", "") or "",
        "output_characters": len(output_text),
        "error": error,
    }


def call_once(
    openai_client: OpenAI | None,
    anthropic_client: Any | None,
    job: Job,
    attempt: int,
    reasoning_effort: str | None,
    allow_cache: bool,
) -> dict[str, Any]:
    if job.provider == "anthropic":
        if anthropic_client is None:
            raise RuntimeError("Anthropic client was not initialized")
        return call_anthropic_once(anthropic_client, job, attempt, allow_cache)
    if openai_client is None:
        raise RuntimeError("OpenAI client was not initialized")
    return call_openai_once(
        openai_client,
        job,
        attempt,
        reasoning_effort,
        allow_cache,
    )


def error_row(job: Job, attempt: int, exc: Exception) -> dict[str, Any]:
    status_code = getattr(exc, "status_code", None)
    message = " ".join(str(exc).split())[:500]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run": job.run,
        "attempt": attempt,
        "provider": job.provider,
        "model": job.model,
        "tier": job.tier_label,
        "requested_service_tier": job.service_tier,
        "actual_service_tier": "",
        "tier_valid": False,
        "scenario": job.scenario.scenario,
        "context_size": job.scenario.context_size,
        "context_tokens_target": job.scenario.context_tokens_target,
        "context_tokens_actual": job.context.tokens_actual,
        "context_start_paragraph": job.context.start_paragraph,
        "context_end_paragraph": job.context.end_paragraph,
        "context_sha256": job.context.sha256,
        "prompt_tokens_estimate": job.prompt_tokens_estimate,
        "reply_size": job.scenario.reply_size,
        "max_output_tokens": job.scenario.max_output_tokens,
        "status": "error",
        "response_status": "",
        "incomplete_reason": "",
        "ttft_seconds": "",
        "full_seconds": "",
        "input_tokens": "",
        "cached_input_tokens": "",
        "output_tokens": "",
        "response_id": "",
        "output_characters": "",
        "error": f"HTTP {status_code}: {message}" if status_code else message,
    }


def should_retry(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code not in {400, 401, 403, 404, 422}


def is_unavailable_tier_error(job: Job, exc: Exception) -> bool:
    """Recognize a model/tier incompatibility or project-tier restriction."""
    if job.service_tier == "default":
        return False
    status_code = getattr(exc, "status_code", None)
    # Streaming API errors may carry the server message without preserving the
    # original HTTP status. Accept an absent status only when the message/body
    # itself identifies service_tier as the failing parameter.
    if status_code is not None and status_code not in {400, 403, 422}:
        return False
    message = str(exc).lower()
    body = getattr(exc, "body", None)
    body_param = body.get("param") if isinstance(body, dict) else None
    exception_param = getattr(exc, "param", None)
    mentions_tier = (
        body_param == "service_tier"
        or exception_param == "service_tier"
        or "service_tier" in message
        or "service tier" in message
        or job.service_tier in message
    )
    mentions_unavailable = any(
        word in message
        for word in ("support", "invalid", "available", "allowed", "permit", "permission", "access")
    )
    return mentions_tier and mentions_unavailable


def unavailable_tier_row(job: Job, reason: str) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run": job.run,
        "attempt": 0,
        "provider": job.provider,
        "model": job.model,
        "tier": job.tier_label,
        "requested_service_tier": job.service_tier,
        "actual_service_tier": "",
        "tier_valid": False,
        "scenario": job.scenario.scenario,
        "context_size": job.scenario.context_size,
        "context_tokens_target": job.scenario.context_tokens_target,
        "context_tokens_actual": job.context.tokens_actual,
        "context_start_paragraph": job.context.start_paragraph,
        "context_end_paragraph": job.context.end_paragraph,
        "context_sha256": job.context.sha256,
        "prompt_tokens_estimate": job.prompt_tokens_estimate,
        "reply_size": job.scenario.reply_size,
        "max_output_tokens": job.scenario.max_output_tokens,
        "status": "unavailable_tier",
        "response_status": "",
        "incomplete_reason": "",
        "ttft_seconds": "",
        "full_seconds": "",
        "input_tokens": "",
        "cached_input_tokens": "",
        "output_tokens": "",
        "response_id": "",
        "output_characters": "",
        "error": f"Skipped after model/tier probe: {reason}",
    }


def append_raw(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RAW_FIELDS})
        handle.flush()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["provider"], row["model"], row["tier"], row["scenario"])
        ].append(row)

    summary: list[dict[str, Any]] = []
    for (provider, model, tier, scenario), attempts in sorted(grouped.items()):
        group = [row for row in attempts if row["status"] == "ok"]
        statuses = {str(row["status"]) for row in attempts}
        if group:
            cell_status = "ok"
        elif "unavailable_tier" in statuses:
            cell_status = "unavailable"
        else:
            cell_status = "failed"
        ttft = [float(row["ttft_seconds"]) for row in group if row["ttft_seconds"] != ""]
        full = [float(row["full_seconds"]) for row in group]
        output_tokens = [
            float(row["output_tokens"]) for row in group if row["output_tokens"] != ""
        ]
        context_tokens = [int(row["context_tokens_actual"]) for row in attempts]
        actual_tiers = Counter(str(row["actual_service_tier"]) for row in group)
        first = attempts[0]
        summary.append(
            {
                "provider": provider,
                "model": model,
                "tier": tier,
                "status": cell_status,
                "actual_tiers": ";".join(f"{key}:{value}" for key, value in sorted(actual_tiers.items())),
                "scenario": scenario,
                "context_size": first["context_size"],
                "context_tokens_target": first["context_tokens_target"],
                "context_tokens_actual_min": min(context_tokens),
                "context_tokens_actual_max": max(context_tokens),
                "reply_size": first["reply_size"],
                "successful_generations": len(group),
                "ttft_samples": len(ttft),
                "ttft_mean_s": statistics.fmean(ttft) if ttft else "",
                "ttft_median_s": statistics.median(ttft) if ttft else "",
                "ttft_p95_s": percentile(ttft, 0.95) if ttft else "",
                "full_mean_s": statistics.fmean(full) if full else "",
                "full_median_s": statistics.median(full) if full else "",
                "full_p95_s": percentile(full, 0.95) if full else "",
                "output_tokens_mean": statistics.fmean(output_tokens) if output_tokens else "",
            }
        )
    return summary


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate successful runs without mixing different scenarios."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            grouped[
                (row["scenario"], row["provider"], row["model"], row["tier"])
            ].append(row)

    aggregate: list[dict[str, Any]] = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            SCENARIO_NAMES.index(item[0][0]),
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    )
    for (scenario, provider, model, tier), group in ordered_groups:
        ttft = [float(row["ttft_seconds"]) for row in group if row["ttft_seconds"] != ""]
        full = [float(row["full_seconds"]) for row in group]
        aggregate.append(
            {
                "scenario": scenario,
                "provider": provider,
                "model": model,
                "tier": tier,
                "successful_generations": len(group),
                "ttft_samples": len(ttft),
                "ttft_min_s": min(ttft) if ttft else "",
                "ttft_mean_s": statistics.fmean(ttft) if ttft else "",
                "ttft_max_s": max(ttft) if ttft else "",
                "full_min_s": min(full) if full else "",
                "full_mean_s": statistics.fmean(full) if full else "",
                "full_max_s": max(full) if full else "",
            }
        )
    return aggregate


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    if not summary:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def write_summary_markdown(
    path: Path,
    summary: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    target_runs: int,
) -> None:
    lines = [
        "# OpenAI and Anthropic latency benchmark",
        "",
        "TTFT is measured from request start to the first non-empty output-text delta. "
        "Full latency is measured from request start through the completed streamed response.",
        "",
        "| Provider | Model | Tier | Scenario | Status | N | Mean TTFT (s) | Mean full (s) | P95 TTFT (s) | P95 full (s) |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['provider']} | {item['model']} | {item['tier']} | {item['scenario']} | "
            f"{item['status']} | {item['successful_generations']} | {fmt(item['ttft_mean_s'])} | "
            f"{fmt(item['full_mean_s'])} | {fmt(item['ttft_p95_s'])} | "
            f"{fmt(item['full_p95_s'])} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate by scenario",
            "",
            "Each scenario is aggregated independently. Only successful generations are included.",
        ]
    )
    for scenario in SCENARIO_NAMES:
        scenario_items = [item for item in aggregate if item["scenario"] == scenario]
        if not scenario_items:
            continue
        lines.extend(
            [
                "",
                f"### `{scenario}`",
                "",
                "| Provider | Model | Tier | N | TTFT min (s) | TTFT mean (s) | TTFT max (s) | Full min (s) | Full mean (s) | Full max (s) |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in scenario_items:
            lines.append(
                f"| {item['provider']} | {item['model']} | {item['tier']} | "
                f"{item['successful_generations']} | "
                f"{fmt(item['ttft_min_s'])} | {fmt(item['ttft_mean_s'])} | "
                f"{fmt(item['ttft_max_s'])} | {fmt(item['full_min_s'])} | "
                f"{fmt(item['full_mean_s'])} | {fmt(item['full_max_s'])} |"
            )

    unavailable = [row for row in rows if row["status"] == "unavailable_tier"]
    invalid = [row for row in rows if row["status"] not in {"ok", "unavailable_tier"}]
    unavailable_cells = [item for item in summary if item["status"] == "unavailable"]
    incomplete = [
        item
        for item in summary
        if item["status"] != "unavailable" and item["successful_generations"] < target_runs
    ]
    if unavailable or invalid or incomplete:
        lines.extend(
            [
                "",
                "## Data-quality notes",
                "",
                f"- Unavailable model/tier cells: {len(unavailable_cells)}.",
                f"- Planned generations skipped for unavailable tiers: {len(unavailable)}.",
                f"- Other invalid attempts recorded: {len(invalid)}.",
                f"- Incomplete cells present in the summary: {len(incomplete)}.",
                "- See `raw_results.csv` for unavailable tiers, errors, and tier mismatches.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "" if value == "" else f"{float(value):.3f}"


def print_progress(index: int, total: int, row: dict[str, Any]) -> None:
    ttft = fmt(row["ttft_seconds"]) if row["ttft_seconds"] != "" else "n/a"
    full = fmt(row["full_seconds"]) if row["full_seconds"] != "" else "n/a"
    print(
        f"[{index:>4}/{total}] {row['provider']:<9} {row['model']} {row['tier']:<9} "
        f"{row['scenario']:<35} {row['status']:<13} TTFT={ttft}s full={full}s",
        flush=True,
    )


def print_aggregate(aggregate: list[dict[str, Any]]) -> None:
    print("\nAggregate by scenario (successful generations only):")
    if not aggregate:
        print("  No successful generations to aggregate.")
        return
    for scenario in SCENARIO_NAMES:
        scenario_items = [item for item in aggregate if item["scenario"] == scenario]
        if not scenario_items:
            continue
        print(f"\n{scenario}")
        print(
            f"{'Provider':<9} {'Model':<27} {'Tier':<9} {'N':>4} "
            f"{'TTFT min':>10} {'mean':>8} {'max':>8} "
            f"{'Full min':>10} {'mean':>8} {'max':>8}"
        )
        for item in scenario_items:
            print(
                f"{item['provider']:<9} {item['model']:<27} {item['tier']:<9} "
                f"{item['successful_generations']:>4} "
                f"{fmt(item['ttft_min_s']):>10} {fmt(item['ttft_mean_s']):>8} "
                f"{fmt(item['ttft_max_s']):>8} {fmt(item['full_min_s']):>10} "
                f"{fmt(item['full_mean_s']):>8} {fmt(item['full_max_s']):>8}"
            )


def write_result_summaries(
    output_dir: Path,
    rows: list[dict[str, Any]],
    target_runs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = summarize(rows)
    aggregate = aggregate_results(rows)
    write_summary_csv(output_dir / "summary.csv", summary)
    write_summary_csv(output_dir / "aggregate.csv", aggregate)
    write_summary_markdown(
        output_dir / "summary.md",
        summary,
        aggregate,
        rows,
        target_runs,
    )
    return summary, aggregate


def job_outcomes(rows: list[dict[str, Any]]) -> Counter[str]:
    grouped: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            row["run"],
            row["provider"],
            row["model"],
            row["tier"],
            row["scenario"],
        )
        grouped[key].add(str(row["status"]))

    outcomes: Counter[str] = Counter()
    for statuses in grouped.values():
        if "ok" in statuses:
            outcomes["ok"] += 1
        elif "unavailable_tier" in statuses:
            outcomes["unavailable_tier"] += 1
        else:
            outcomes["failed"] += 1
    return outcomes


def main() -> int:
    args = parse_args()
    if args.list_models:
        for model in ALL_TEXT_MODELS:
            print(f"{provider_for_model(model):<9} {model}")
        return 0
    if args.list_model_tiers:
        for model in ALL_TEXT_MODELS:
            print(
                f"{provider_for_model(model):<9} {model:<27} "
                f"{', '.join(supported_tier_labels(model))}"
            )
        return 0
    if args.list_scenarios:
        print("\n".join(SCENARIO_NAMES))
        return 0
    if args.show_prompt:
        enc = encoding()
        try:
            corpus = load_corpus(args.corpus_file, args.corpus_url)
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"Error loading corpus: {exc}", file=sys.stderr)
            return 2
        scenario = build_scenarios([args.show_prompt])[0]
        sample_rng = random.Random(
            f"{args.seed}:1:{scenario.context_size}:{corpus.sha256}"
        )
        sample = sample_context(
            corpus,
            scenario.context_tokens_target,
            enc,
            sample_rng,
        )
        prompt = render_prompt(sample.text, scenario.response_instruction)
        if args.allow_cache:
            print(prompt)
        else:
            print("Benchmark sample ID: <random UUID generated per request>\n")
            print(prompt)
        return 0

    models = selected_models(args)
    enc = encoding()
    try:
        corpus = load_corpus(args.corpus_file, args.corpus_url)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Error loading corpus: {exc}", file=sys.stderr)
        return 2
    scenarios = build_scenarios(args.scenarios)
    tiers = normalize_tiers(args.tiers)
    skipped_pairs = unsupported_model_tiers(models, tiers)
    jobs = build_jobs(models, tiers, scenarios, args.runs, args.seed, corpus, enc)

    output_dir = args.output_dir or Path("benchmark-results") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_selection": (
            "all"
            if args.all_models
            else "openai"
            if args.all_openai_models
            else "anthropic"
            if args.all_anthropic_models
            else "explicit"
        ),
        "model_catalog_as_of": MODEL_CATALOG_AS_OF,
        "model_tier_support_as_of": MODEL_TIER_SUPPORT_AS_OF,
        "models": models,
        "model_providers": {model: provider_for_model(model) for model in models},
        "tiers": [{"label": label, "service_tier": tier} for label, tier in tiers],
        "supported_tiers_by_model": {
            model: list(supported_tier_labels(model)) for model in models
        },
        "skipped_unsupported_model_tiers": [
            {"model": model, "tier": tier} for model, tier in skipped_pairs
        ],
        "runs": args.runs,
        "reasoning_effort": args.reasoning_effort,
        "allow_cache": args.allow_cache,
        "sequential_requests": True,
        "seed": args.seed,
        "corpus": {
            "title": corpus.title,
            "source": corpus.source,
            "sha256": corpus.sha256,
            "paragraphs": len(corpus.paragraphs),
        },
        "context_sampling": {
            "method": "random contiguous paragraph-aligned subsection",
            "shared_across_models_tiers_and_reply_sizes_within_run": True,
            "samples": context_sample_manifest(jobs),
        },
        "pause_seconds": args.pause,
        "timeout_seconds_per_attempt": args.timeout,
        "hard_wall_clock_timeout_available": (
            hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
        ),
        "max_attempts": args.max_attempts,
        "planned_matrix_generations": len(jobs),
        "scenarios": [
            {key: value for key, value in asdict(scenario).items() if key != "prompt"}
            for scenario in scenarios
        ],
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Output: {output_dir.resolve()}")
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Corpus: {corpus.title} ({len(corpus.paragraphs)} paragraphs)")
    print(f"Planned matrix generations: {len(jobs)}")
    unknown_models = [model for model in models if model not in MODEL_TIER_SUPPORT]
    if unknown_models:
        print(
            "Unknown model IDs are conservatively treated as normal-only: "
            + ", ".join(unknown_models)
        )
    for scenario in scenarios:
        scenario_jobs = [job for job in jobs if job.scenario.scenario == scenario.scenario]
        context_tokens = sorted({job.context.tokens_actual for job in scenario_jobs})
        context_actual = (
            str(context_tokens[0])
            if len(context_tokens) == 1
            else f"{context_tokens[0]}-{context_tokens[-1]}"
            if context_tokens
            else "n/a"
        )
        print(
            f"  {scenario.scenario:<35} context_length={context_actual:>9} tokens"
        )

    if args.dry_run:
        print("Dry run complete; no API requests were made.")
        return 0

    if not jobs:
        print("No supported model/tier combinations were selected; no API requests were made.")
        _summary, aggregate = write_result_summaries(output_dir, [], args.runs)
        print_aggregate(aggregate)
        print(f"Summary: {(output_dir / 'summary.md').resolve()}")
        return 0

    selected_providers = {job.provider for job in jobs}
    missing_keys = []
    if "openai" in selected_providers and not os.environ.get("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if "anthropic" in selected_providers and not os.environ.get("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    if missing_keys:
        print(
            "Error: set " + " and ".join(missing_keys) + " before running the benchmark.",
            file=sys.stderr,
        )
        return 2

    openai_client: OpenAI | None = None
    anthropic_client: Any | None = None
    if "openai" in selected_providers:
        openai_client = OpenAI(max_retries=0, timeout=args.timeout)
    if "anthropic" in selected_providers:
        try:
            from anthropic import Anthropic
        except ImportError:
            print(
                "Error: Anthropic support requires the 'anthropic' package; "
                "run: python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 2
        anthropic_client = Anthropic(max_retries=0, timeout=args.timeout)
    raw_path = output_dir / "raw_results.csv"
    rows: list[dict[str, Any]] = []
    unavailable_tiers: dict[tuple[str, str, str], str] = {}

    for job_index, job in enumerate(jobs, start=1):
        tier_key = (job.provider, job.model, job.service_tier)
        if tier_key in unavailable_tiers:
            row = unavailable_tier_row(job, unavailable_tiers[tier_key])
            append_raw(raw_path, row)
            rows.append(row)
            print_progress(job_index, len(jobs), row)
            continue

        valid_row: dict[str, Any] | None = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                with generation_deadline(args.timeout):
                    row = call_once(
                        openai_client,
                        anthropic_client,
                        job,
                        attempt,
                        args.reasoning_effort,
                        args.allow_cache,
                    )
                append_raw(raw_path, row)
                rows.append(row)
                if row["status"] == "ok":
                    valid_row = row
                    break
            except KeyboardInterrupt:
                print("\nInterrupted. Writing summaries for completed requests...", file=sys.stderr)
                _summary, aggregate = write_result_summaries(output_dir, rows, args.runs)
                print_aggregate(aggregate)
                return 130
            except Exception as exc:  # SDK exceptions share status_code but not one base in all versions.
                row = error_row(job, attempt, exc)
                if is_unavailable_tier_error(job, exc):
                    row["status"] = "unavailable_tier"
                    unavailable_tiers[tier_key] = row["error"]
                append_raw(raw_path, row)
                rows.append(row)
                if row["status"] == "unavailable_tier":
                    break
                if not should_retry(exc):
                    break

            if attempt < args.max_attempts:
                print(
                    f"  retrying {job.provider}/{job.model} {job.tier_label} "
                    f"{job.scenario.scenario} after attempt {attempt}/{args.max_attempts}: "
                    f"{row['error'] or row['status']}",
                    flush=True,
                )
                time.sleep(min(2 ** (attempt - 1), 8) + random.random() * 0.25)

        display_row = valid_row or rows[-1]
        print_progress(job_index, len(jobs), display_row)
        if args.pause and job_index < len(jobs):
            time.sleep(args.pause)

    _summary, aggregate = write_result_summaries(output_dir, rows, args.runs)
    outcomes = job_outcomes(rows)
    print(
        f"Finished: {outcomes['ok']} valid, {outcomes['unavailable_tier']} unavailable-tier, "
        f"and {outcomes['failed']} failed matrix generations."
    )
    print_aggregate(aggregate)
    print(f"Summary: {(output_dir / 'summary.md').resolve()}")
    return 0 if outcomes["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
