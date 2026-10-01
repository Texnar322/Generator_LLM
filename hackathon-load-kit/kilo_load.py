#!/usr/bin/env python3
"""Нагрузка OpenAI-совместимой ручки в форме запросов Kilo Code."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import signal
import ssl
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

LOGGER = logging.getLogger("kilo_load")

KIT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATES_DIR = KIT_ROOT / "templates"
TOKEN_ESTIMATE_CHARS = 4.0
DEFAULT_CONTEXT_CALIBRATION = 1.15
KILO_USER_AGENT = "Kilo-Code/7.3.0 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.13"
KILO_REFERER = "https://kilocode.ai"
KILO_TITLE = "Kilo Code"
KILO_ASSISTANT_CONTENT = "\n\n\n"
DEFAULT_MAX_TOKENS = 32_000
DEFAULT_TOP_P = 0.95
DEFAULT_SLO_TTFT_CONTINUATION_SECONDS = 30.0
DEFAULT_SLO_TTFT_ENTRY_SECONDS = 120.0
DEFAULT_SLO_TOKENS_PER_SECOND = 50.0
DEFAULT_SLO_ERROR_RATE = 0.01
DEFAULT_BURST_TURNS = 8
REDACT_HEADER_NAMES = {"authorization", "proxy-authorization", "cookie", "set-cookie"}
EXPECTED_TOOL_NAMES = (
    "bash",
    "read",
    "glob",
    "grep",
    "edit",
    "write",
    "task",
    "webfetch",
    "todowrite",
    "skill",
    "suggest",
    "kilo_local_recall",
)
EXCLUDED_CORPUS_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".playwright-mcp",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "env",
    "huggingface",
    "node_modules",
    "qdrant_data",
    "results",
    "site-packages",
    "target",
    "venv",
    "vendor",
}
SOURCE_FILE_SUFFIXES = {
    ".c",
    ".cc",
    ".css",
    ".cs",
    ".dockerfile",
    ".example",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".j2",
    ".json",
    ".jsonl",
    ".kt",
    ".kts",
    ".local",
    ".md",
    ".ndjson",
    ".php",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".scss",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".ts",
    ".tsx",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}
SOURCE_FILE_NAMES = {"BUILD", "Dockerfile", "Jenkinsfile", "Makefile"}
TOOL_RESULT_SIZE_BUCKETS: tuple[tuple[float, int, int], ...] = (
    (54.0, 1, 100),
    (28.0, 100, 500),
    (13.0, 500, 2000),
    (4.3, 2000, 8000),
    (0.1, 8000, 16000),
)
HUMAN_PAUSE_QUANTILES: tuple[tuple[float, float], ...] = (
    (0.25, 35.5),
    (0.50, 133.9),
    (0.75, 386.4),
    (0.95, 1374.3),
)
# Пауза агента внутри всплеска (между ходами), не человеческая.
AGENT_THINK_SECONDS: tuple[float, ...] = (0.05, 0.08, 0.12, 0.18, 0.27, 0.41)
TURN_PROMPT_TEMPLATES: tuple[str, ...] = (
    "Review the function shown in the tool output above, point out the exact bug or missing "
    "edge case, and rewrite it correctly.",
    "Implement the missing piece indicated by the code above: write the full function body, "
    "matching the existing style.",
    "Refactor the module shown above to remove duplication without changing its public API; "
    "show the resulting code.",
    "Explain what the code above actually does and propose a concrete fix for the most likely "
    "defect in it.",
    "Add proper error handling and type hints to the function shown above and show the "
    "corrected version.",
    "Write a unit test that reproduces the edge case exposed by the code above, then fix the "
    "code so it passes.",
)
INITIAL_TASK = (
    "Explore the codebase in the workspace, identify a non-trivial module, and implement a "
    "focused correctness fix plus a regression test. Use the available tools; do not ask "
    "follow-up questions."
)


@dataclass
class VirtualSession:
    session_id: str
    affinity: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    model: str
    max_tokens: int
    corpus_slot: int = 0
    turns: int = 0
    context_characters: int | None = None
    context_message_count: int = 0
    context_calibration: float | None = None


@dataclass
class StreamObservation:
    status_code: int | None
    ttft_seconds: float | None
    last_token_seconds: float | None = None
    inter_token_seconds: list[float] = field(default_factory=list)
    e2e_seconds: float | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    output_text: str = ""
    error: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет общие флаги URL, модели, ключа и корпуса."""
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, оканчивается на /v1")
    parser.add_argument("--model", required=True, help="Имя модели в теле запроса")
    parser.add_argument("--api-key-env", required=True, help="Имя env-переменной с bearer token")
    parser.add_argument("--corpus-root", type=Path, required=True, help="Каталог с исходным кодом для контекста")
    parser.add_argument("--output", type=Path, required=True, help="JSONL наблюдений")
    parser.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR)
    parser.add_argument("--ca-file", action="append", type=Path, default=[], help="PEM CA; флаг можно повторять")
    parser.add_argument("--sessions", type=int, default=300)
    parser.add_argument("--target-context-tokens", type=int, default=200_000)
    parser.add_argument("--growth-tokens-per-turn", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--burst-turns", type=int, default=DEFAULT_BURST_TURNS)
    parser.add_argument("--session-start-jitter-seconds", type=float, default=None)
    parser.add_argument("--session-ramp-seconds", type=float, default=None)
    parser.add_argument("--slo-ttft-continuation-seconds", type=float, default=DEFAULT_SLO_TTFT_CONTINUATION_SECONDS)
    parser.add_argument("--slo-ttft-entry-seconds", type=float, default=DEFAULT_SLO_TTFT_ENTRY_SECONDS)
    parser.add_argument("--slo-tokens-per-second", type=float, default=DEFAULT_SLO_TOKENS_PER_SECOND)
    parser.add_argument("--slo-error-rate", type=float, default=DEFAULT_SLO_ERROR_RATE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает CLI: сценарий и общие параметры нагрузки."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="scenario", required=True)

    steps = sub.add_parser("steps", help="Ступени одновременных сессий")
    add_common_arguments(steps)
    steps.add_argument("--steps", help="Список N через запятую; иначе 1,2,4,...,sessions")
    steps.add_argument("--step-duration-seconds", type=float, default=300.0)
    steps.add_argument("--cooldown-seconds", type=float, default=10.0)

    burst = sub.add_parser("burst", help="Почти одновременный старт N сессий")
    add_common_arguments(burst)
    burst.add_argument("--duration-seconds", type=float, default=600.0)

    soak = sub.add_parser("soak", help="Удержание N сессий заданную длительность")
    add_common_arguments(soak)
    soak.add_argument("--duration-seconds", type=float, default=2500)
    soak.add_argument("--duration-hours", type=float, default=None)
    return parser.parse_args(argv)


def apply_scenario_defaults(args: argparse.Namespace) -> None:
    """Подставляет джиттер/рамп/длительность, если их не задали явно."""
    if args.scenario == "burst":
        if args.session_start_jitter_seconds is None:
            args.session_start_jitter_seconds = 1.0
        if args.session_ramp_seconds is None:
            args.session_ramp_seconds = 0.0
        args.wave_duration_seconds = args.duration_seconds
    elif args.scenario == "steps":
        if args.session_start_jitter_seconds is None:
            args.session_start_jitter_seconds = 15.0
        if args.session_ramp_seconds is None:
            args.session_ramp_seconds = 120.0
        args.wave_duration_seconds = args.step_duration_seconds
    else:
        if args.session_start_jitter_seconds is None:
            args.session_start_jitter_seconds = 60.0
        if args.session_ramp_seconds is None:
            args.session_ramp_seconds = 60.0
        if args.duration_seconds is not None and args.duration_hours is not None:
            raise ValueError("укажите либо --duration-seconds, либо --duration-hours")
        if args.duration_seconds is not None:
            args.wave_duration_seconds = args.duration_seconds
        elif args.duration_hours is not None:
            args.wave_duration_seconds = args.duration_hours * 3600.0
        else:
            args.wave_duration_seconds = 3.0 * 3600.0


def canonical_json(value: Any) -> str:
    """Сериализует JSON канонически для оценки размера и сравнения префикса."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def percentile(values: list[float], quantile: float) -> float | None:
    """Линейно интерполирует квантиль; пустой список -> None."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def describe(values: list[float]) -> dict[str, float | int | None]:
    """Краткая сводка распределения."""
    return {
        "count": len(values),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def load_templates(templates_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    """Читает system prompt и схемы инструментов Kilo Code."""
    system_path = templates_dir / "system_prompt.txt"
    tools_path = templates_dir / "tools.json"
    system_prompt = system_path.read_text(encoding="utf-8")
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"В {tools_path} нет списка инструментов")
    names = tuple(
        (tool.get("function") or {}).get("name") for tool in tools if isinstance(tool, dict)
    )
    if names != EXPECTED_TOOL_NAMES:
        raise ValueError(f"Неожиданный набор инструментов: {names}")
    return system_prompt, tools


def user_message(*texts: str) -> dict[str, Any]:
    """Собирает user-сообщение Kilo: массив text-частей."""
    return {"role": "user", "content": [{"type": "text", "text": text} for text in texts]}


def build_system_content(template: str, model: str, workdir: str, marker: str) -> str:
    """Дописывает к промпту Kilo env-блок и уникальный маркер сессии."""
    today = datetime.now(timezone.utc).strftime("%a %b %d %Y")
    return (
        template.rstrip()
        + "\n\n"
        + f"You are powered by the model named {model}. The exact model ID is {model}\n"
        + "Here is some useful information about the environment you are running in:\n"
        + "<env>\n"
        + "  Is directory a git repo: yes\n"
        + "  Platform: linux\n"
        + f"  Today's date: {today}\n"
        + f"  Working directory: {workdir}\n"
        + f"  Workspace root folder: {workdir}\n"
        + "</env>\n"
        + f"<load-test-session>{marker}</load-test-session>\n"
    )


def environment_details(workdir: str) -> str:
    """Вторая text-часть первого user-сообщения в форме Kilo."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return (
        "<environment_details>\n"
        f"Current time: {now}\n"
        f"Working directory: {workdir}\n"
        f"Workspace root folder: {workdir}\n"
        "</environment_details>"
    )


def create_session(
    index: int,
    model: str,
    tools: list[dict[str, Any]],
    system_template: str,
    workdir: str,
    max_tokens: int,
) -> VirtualSession:
    """Создаёт сессию с уникальным маркером, affinity и расходящимся префиксом."""
    marker = f"load-test-session={index}-{uuid.uuid4()}"
    affinity = f"ses_{uuid.uuid4().hex}"
    system = {
        "role": "system",
        "content": build_system_content(system_template, model, workdir, marker),
    }
    first_user = user_message(
        f"{INITIAL_TASK} (session={marker})",
        environment_details(workdir),
    )
    return VirtualSession(
        session_id=marker,
        affinity=affinity,
        messages=[system, first_user],
        tools=tools,
        model=model,
        max_tokens=max_tokens,
        corpus_slot=index,
    )


def session_headers(affinity: str, api_key: str) -> dict[str, str]:
    """Заголовки Kilo Code; affinity постоянен внутри сессии."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": KILO_USER_AGENT,
        "http-referer": KILO_REFERER,
        "x-title": KILO_TITLE,
        "x-session-affinity": affinity,
    }


def estimate_context_tokens(session: VirtualSession) -> int:
    """Оценивает длину контекста по JSON сообщений и tools."""
    if session.context_characters is None or session.context_message_count != len(session.messages):
        empty_request = {"messages": [], "tools": session.tools, "model": session.model}
        session.context_characters = len(canonical_json(empty_request)) + max(0, len(session.messages) - 1)
        session.context_characters += sum(len(canonical_json(message)) for message in session.messages)
        session.context_message_count = len(session.messages)
    return math.ceil(session.context_characters / TOKEN_ESTIMATE_CHARS)


def calibrated_context_tokens(session: VirtualSession) -> int:
    """Оценка контекста с калибровочным коэффициентом."""
    estimate = estimate_context_tokens(session)
    calibration = DEFAULT_CONTEXT_CALIBRATION if session.context_calibration is None else session.context_calibration
    return math.ceil(estimate * calibration)


def append_messages(session: VirtualSession, messages: list[dict[str, Any]]) -> None:
    """Дописывает сообщения и поддерживает оценку длины."""
    estimate_context_tokens(session)
    previous_count = len(session.messages)
    session.messages.extend(messages)
    assert session.context_characters is not None
    session.context_characters += sum(len(canonical_json(message)) for message in messages)
    session.context_characters += max(0, len(session.messages) - 1) - max(0, previous_count - 1)
    session.context_message_count = len(session.messages)


def calibrate_context(session: VirtualSession, estimate: int, actual: int | None) -> None:
    """Подкручивает калибровку по prompt_tokens ответа."""
    if actual is None or actual <= 0 or estimate <= 0:
        return
    ratio = min(2.0, max(0.5, actual / estimate))
    previous = session.context_calibration
    session.context_calibration = ratio if previous is None else (previous + ratio) / 2


def corpus_files(corpus_root: Path) -> list[Path]:
    """Собирает текстовые исходники корпуса, пропуская служебные каталоги."""
    if not corpus_root.is_dir():
        raise NotADirectoryError(corpus_root)
    files: list[Path] = []
    for root, directories, filenames in os.walk(corpus_root):
        directories[:] = sorted(
            directory for directory in directories if directory not in EXCLUDED_CORPUS_DIRECTORIES
        )
        root_path = Path(root)
        files.extend(
            root_path / filename
            for filename in sorted(filenames)
            if (root_path / filename).suffix in SOURCE_FILE_SUFFIXES or filename in SOURCE_FILE_NAMES
        )
    if not files:
        raise ValueError(f"В корпусе {corpus_root} нет текстовых исходников")
    return files


@lru_cache(maxsize=None)
def cached_file_text(path: Path) -> str:
    """Читает файл корпуса один раз за процесс."""
    return path.read_text(encoding="utf-8", errors="replace")


def synthetic_tool_result(session: VirtualSession, tokens: int, files: list[Path]) -> str:
    """Набирает результат инструмента из реальных файлов, разные сессии — с разных мест."""
    required_characters = math.ceil(tokens * TOKEN_ESTIMATE_CHARS)
    fragments: list[str] = []
    fragment_length = 0
    file_index = (session.corpus_slot * 1543 + session.turns * 131) % len(files)
    files_without_content = 0
    while fragment_length < required_characters:
        path = files[file_index]
        content = cached_file_text(path)
        if content:
            files_without_content = 0
            remaining = required_characters - fragment_length
            chunk = f"<path>{path}</path>\n<type>file</type>\n<content>\n{content}\n</content>\n"
            fragments.append(chunk[:remaining])
            fragment_length += min(len(chunk), remaining)
            if fragment_length >= required_characters:
                return "".join(fragments)
        else:
            files_without_content += 1
            if files_without_content == len(files):
                raise ValueError("В корпусе нет непустых текстовых исходников")
        file_index = (file_index + 1) % len(files)
    return "".join(fragments)


def append_tool_cycle(session: VirtualSession, tool_result_tokens: int, user_text: str, files: list[Path]) -> None:
    """Дописывает вызов read, результат из корпуса и следующую реплику пользователя."""
    call_id = f"call_{uuid.uuid4().hex[:24]}"
    file_index = (session.corpus_slot * 1543 + session.turns * 131) % len(files)
    file_path = str(files[file_index])
    arguments = json.dumps({"filePath": file_path}, ensure_ascii=False, separators=(",", ":"))
    append_messages(
        session,
        [
            {
                "role": "assistant",
                "content": KILO_ASSISTANT_CONTENT,
                "reasoning_content": f"Inspect {file_path} for {session.session_id}.",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "read", "arguments": arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": synthetic_tool_result(session, tool_result_tokens, files),
            },
            user_message(user_text),
        ],
    )


def grow_history(session: VirtualSession, target_tokens: int, growth_tokens: int, files: list[Path]) -> None:
    """Растит историю одним циклом инструмента до целевой длины."""
    current_tokens = calibrated_context_tokens(session)
    if current_tokens >= target_tokens:
        return
    addition_tokens = min(growth_tokens, target_tokens - current_tokens)
    append_tool_cycle(
        session,
        addition_tokens,
        f"Continue analysis for {session.session_id}; use the previous tool result as context.",
        files,
    )


def prepare_session(session: VirtualSession, target_tokens: int, growth_tokens: int, files: list[Path]) -> None:
    """Набивает сессию реальным кодом до целевого контекста до первого HTTP-запроса."""
    while calibrated_context_tokens(session) < target_tokens:
        grow_history(session, target_tokens, growth_tokens, files)


def sample_tool_result_tokens(randomizer: random.Random) -> int:
    """Сэмплирует размер результата инструмента по бакетам TOOL_RESULT_SIZE_BUCKETS."""
    weights = [weight for weight, _, _ in TOOL_RESULT_SIZE_BUCKETS]
    _, lower_bound, upper_bound = randomizer.choices(TOOL_RESULT_SIZE_BUCKETS, weights=weights, k=1)[0]
    return randomizer.randint(lower_bound, upper_bound)


def sample_human_pause_seconds(randomizer: random.Random) -> float:
    """Сэмплирует паузу человека между всплесками по квантилям HUMAN_PAUSE_QUANTILES."""
    quantile = randomizer.random()
    points = HUMAN_PAUSE_QUANTILES
    if quantile <= points[0][0]:
        (q_far, v_far), (q_near, v_near) = points[1], points[0]
    elif quantile >= points[-1][0]:
        (q_far, v_far), (q_near, v_near) = points[-2], points[-1]
    else:
        for (q_lo, v_lo), (q_hi, v_hi) in zip(points, points[1:]):
            if q_lo <= quantile <= q_hi:
                fraction = (quantile - q_lo) / (q_hi - q_lo)
                return math.exp(math.log(v_lo) + fraction * (math.log(v_hi) - math.log(v_lo)))
        raise AssertionError("quantile вне диапазона [0, 1]")
    slope = math.log(v_near / v_far) / (q_near - q_far)
    return math.exp(math.log(v_near) + slope * (quantile - q_near))


def turn_prompt(session: VirtualSession, turn: int) -> str:
    """Детерминированная по номеру хода задача над кодом из контекста."""
    template = TURN_PROMPT_TEMPLATES[turn % len(TURN_PROMPT_TEMPLATES)]
    return f"{template} (session={session.session_id}, turn={turn})"


def next_pause_seconds(turn: int, burst_turns: int, randomizer: random.Random) -> float:
    """Пауза перед следующим ходом: раз в burst_turns — человек, иначе шаг агента."""
    if turn % burst_turns == 0:
        return sample_human_pause_seconds(randomizer)
    return randomizer.choice(AGENT_THINK_SECONDS)


def request_payload(session: VirtualSession) -> dict[str, Any]:
    """Тело chat/completions в форме Kilo Code (без temperature)."""
    return {
        "model": session.model,
        "max_tokens": session.max_tokens,
        "top_p": DEFAULT_TOP_P,
        "messages": session.messages,
        "tools": session.tools,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def generation_tokens_per_second(
    output_tokens: int | None,
    e2e_seconds: float | None,
    ttft_seconds: float | None,
    last_token_seconds: float | None = None,
) -> float | None:
    """Скорость генерации после TTFT."""
    if output_tokens is None or e2e_seconds is None or ttft_seconds is None:
        return None
    generation_end = e2e_seconds if last_token_seconds is None else last_token_seconds
    generation_seconds = generation_end - ttft_seconds
    if generation_seconds <= 0:
        return None
    return output_tokens / generation_seconds


def session_start_delay_seconds(
    session_index: int,
    sessions: int,
    session_ramp_seconds: float,
    start_jitter_seconds: float,
) -> float:
    """Задержка старта сессии: рамп по индексу плюс джиттер."""
    spread = 0.0 if sessions <= 1 else (session_index - 1) * session_ramp_seconds / sessions
    return spread + start_jitter_seconds


def public_headers(headers: dict[str, str]) -> dict[str, str]:
    """Копия заголовков без секретов."""
    return {
        key: ("<redacted>" if key.lower() in REDACT_HEADER_NAMES else value)
        for key, value in headers.items()
    }


async def sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Итератор SSE-событий chat/completions."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload:
            continue
        if payload == "[DONE]":
            yield {"_stream_done": True}
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            yield {"_stream_error": "malformed_sse_event"}
            continue
        if isinstance(event, dict):
            yield event


def delta_text(event: dict[str, Any]) -> str:
    """Достаёт текст/reasoning/аргументы tool_calls из delta."""
    choices = event.get("choices", [])
    if not choices:
        return ""
    delta = choices[0].get("delta", {})
    fragments = [
        value for key, value in delta.items() if key in {"content", "reasoning_content"} and isinstance(value, str)
    ]
    for call in delta.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function", {})
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            fragments.append(arguments)
    return "".join(fragments)


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> StreamObservation:
    """POST /chat/completions со стримом; считает TTFT и usage."""
    started_at = time.perf_counter()
    observation = StreamObservation(status_code=None, ttft_seconds=None)
    first_token_at: float | None = None
    previous_token_at: float | None = None
    stream_done = False
    malformed_stream = False
    try:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            observation.status_code = response.status_code
            observation.response_headers = {key: value for key, value in response.headers.items()}
            if response.is_error:
                body = (await response.aread())[:2000].decode(errors="replace")
                observation.error = f"HTTP {response.status_code}: {body}"
                return observation
            async for event in sse_events(response):
                if event.get("_stream_done"):
                    stream_done = True
                    continue
                if event.get("_stream_error"):
                    malformed_stream = True
                    continue
                now = time.perf_counter()
                token = delta_text(event)
                if token:
                    observation.output_text += token
                    if first_token_at is None:
                        first_token_at = now
                        observation.ttft_seconds = now - started_at
                    if previous_token_at is not None:
                        observation.inter_token_seconds.append(now - previous_token_at)
                    previous_token_at = now
                    observation.last_token_seconds = now - started_at
                usage = event.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    prompt_tokens_details = usage.get("prompt_tokens_details")
                    if isinstance(prompt_tokens, int):
                        observation.prompt_tokens = prompt_tokens
                    if isinstance(completion_tokens, int):
                        observation.output_tokens = completion_tokens
                    if isinstance(prompt_tokens_details, dict):
                        cached_tokens = prompt_tokens_details.get("cached_tokens")
                        if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool):
                            observation.cached_tokens = cached_tokens
        if observation.output_tokens is None and observation.output_text:
            observation.output_tokens = math.ceil(len(observation.output_text) / 4)
        if malformed_stream:
            observation.error = "malformed_sse_event"
        elif not stream_done:
            observation.error = "incomplete_stream"
    except (httpx.HTTPError, asyncio.TimeoutError, ssl.SSLError) as error:
        observation.error = f"{type(error).__name__}: {error}"
    finally:
        observation.e2e_seconds = time.perf_counter() - started_at
    return observation


def log_http_error(observation: StreamObservation) -> None:
    """Пишет код, тело и публичные заголовки ошибки, без секретов."""
    LOGGER.error(
        "http_error status=%s headers=%s body=%s",
        observation.status_code,
        public_headers(observation.response_headers),
        observation.error,
    )


async def run_session(
    session: VirtualSession,
    session_index: int,
    sessions: int,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    api_key: str,
    files: list[Path],
    deadline: float,
    stop: asyncio.Event,
    records: list[dict[str, Any]],
    write_lock: asyncio.Lock,
    output_stream: Any,
    step_started_at: float,
    step_sessions: int,
) -> None:
    """Ведёт одну сессию: разнесённый вход, всплески ходов, паузы человека."""
    randomizer = random.Random(session.session_id)
    start_jitter_seconds = randomizer.uniform(0.0, args.session_start_jitter_seconds)
    start_delay_seconds = session_start_delay_seconds(
        session_index,
        sessions,
        args.session_ramp_seconds,
        start_jitter_seconds,
    )
    remaining_before_start = deadline - time.monotonic()
    if remaining_before_start <= 0:
        return
    try:
        await asyncio.wait_for(stop.wait(), timeout=min(start_delay_seconds, remaining_before_start))
    except asyncio.TimeoutError:
        pass
    if stop.is_set() or time.monotonic() >= deadline:
        return
    prepare_session(session, args.target_context_tokens, args.growth_tokens_per_turn, files)
    headers = session_headers(session.affinity, api_key)
    turn = 0
    while not stop.is_set() and time.monotonic() < deadline:
        payload = request_payload(session)
        context_estimate = estimate_context_tokens(session)
        start_offset_seconds = time.monotonic() - step_started_at
        observation = await send_request(client, args.base_url, headers, payload)
        if observation.error is not None and observation.status_code is not None:
            log_http_error(observation)
        successful = observation.error is None and bool(observation.output_text)
        turn += 1
        if successful:
            calibrate_context(session, context_estimate, observation.prompt_tokens)
            append_messages(session, [{"role": "assistant", "content": observation.output_text}])
            session.turns += 1
            append_tool_cycle(
                session,
                sample_tool_result_tokens(randomizer),
                turn_prompt(session, turn),
                files,
            )
        think_after_seconds = next_pause_seconds(turn, args.burst_turns, randomizer)
        record = {
            "session_id": session.session_id,
            "affinity": session.affinity,
            "step_sessions": step_sessions,
            "turn": turn,
            "context_tokens_estimate": context_estimate,
            "context_tokens_actual": observation.prompt_tokens,
            "ttft_seconds": observation.ttft_seconds,
            "inter_token_seconds": observation.inter_token_seconds,
            "e2e_seconds": observation.e2e_seconds,
            "output_tokens": observation.output_tokens,
            "cached_tokens": observation.cached_tokens,
            "generation_tokens_per_second": generation_tokens_per_second(
                observation.output_tokens,
                observation.e2e_seconds,
                observation.ttft_seconds,
                observation.last_token_seconds,
            ),
            "status_code": observation.status_code,
            "error": observation.error,
            "empty_completion": not observation.output_text,
            "is_entry": turn == 1,
            "start_offset_seconds": start_offset_seconds,
            "response_headers": public_headers(observation.response_headers) if observation.error else {},
            "think_after_seconds": think_after_seconds,
        }
        async with write_lock:
            records.append(record)
            output_stream.write(canonical_json(record) + "\n")
            output_stream.flush()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=min(think_after_seconds, remaining))
        except asyncio.TimeoutError:
            pass


def build_ssl_context(ca_files: list[Path]) -> ssl.SSLContext:
    """Системное хранилище плюс опциональные CA-бандлы."""
    ssl_context = ssl.create_default_context()
    for ca_file in ca_files:
        if not ca_file.is_file():
            raise FileNotFoundError(ca_file)
        ssl_context.load_verify_locations(cafile=ca_file)
    return ssl_context


async def run_wave(
    args: argparse.Namespace,
    sessions_count: int,
    duration_seconds: float,
    api_key: str,
    system_template: str,
    tools: list[dict[str, Any]],
    files: list[Path],
    output_stream: Any,
    records: list[dict[str, Any]],
    write_lock: asyncio.Lock,
    stop: asyncio.Event,
) -> None:
    """Запускает одну волну из sessions_count параллельных сессий."""
    sessions = [
        create_session(
            index=index + 1,
            model=args.model,
            tools=tools,
            system_template=system_template,
            workdir=str(args.corpus_root),
            max_tokens=args.max_tokens,
        )
        for index in range(sessions_count)
    ]
    timeout = httpx.Timeout(args.request_timeout_seconds, connect=30.0)
    ssl_context = build_ssl_context(args.ca_file)
    step_started_at = time.monotonic()
    deadline = step_started_at + duration_seconds
    async with httpx.AsyncClient(timeout=timeout, http2=False, verify=ssl_context) as client:
        tasks = [
            asyncio.create_task(
                run_session(
                    session,
                    index + 1,
                    sessions_count,
                    client,
                    args,
                    api_key,
                    files,
                    deadline,
                    stop,
                    records,
                    write_lock,
                    output_stream,
                    step_started_at,
                    sessions_count,
                )
            )
            for index, session in enumerate(sessions)
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def step_session_counts(max_sessions: int, raw: str | None) -> list[int]:
    """Список размеров ступеней: явный или геометрический до максимума."""
    if raw:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if not values or min(values) < 1:
            raise ValueError("список --steps должен содержать целые >= 1")
        return values
    counts: list[int] = []
    size = 1
    while size < max_sessions:
        counts.append(size)
        size *= 2
    counts.append(max_sessions)
    return counts


def error_count(records: list[dict[str, Any]]) -> int:
    """Число отказов: HTTP/сеть/парсер или пустой completion."""
    return sum(1 for record in records if record.get("error") is not None or record.get("empty_completion"))


def sla_block(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Считает метрики волны и вердикт SLA."""
    requests = len(records)
    errors = error_count(records)
    error_rate = errors / requests if requests else 1.0
    http_codes: dict[str, int] = {}
    for record in records:
        code = record.get("status_code")
        key = "none" if code is None else str(code)
        http_codes[key] = http_codes.get(key, 0) + 1
    entry = [record for record in records if record.get("is_entry")]
    continuation = [record for record in records if not record.get("is_entry")]
    entry_ttft = [record["ttft_seconds"] for record in entry if isinstance(record.get("ttft_seconds"), (int, float))]
    continuation_ttft = [
        record["ttft_seconds"] for record in continuation if isinstance(record.get("ttft_seconds"), (int, float))
    ]
    speeds = [
        record["generation_tokens_per_second"]
        for record in records
        if isinstance(record.get("generation_tokens_per_second"), (int, float))
    ]
    entry_p95 = percentile(entry_ttft, 0.95)
    continuation_p95 = percentile(continuation_ttft, 0.95)
    speed_p5 = percentile(speeds, 0.05)
    entry_ok = entry_p95 is not None and entry_p95 <= args.slo_ttft_entry_seconds
    if continuation:
        continuation_ok = continuation_p95 is not None and continuation_p95 <= args.slo_ttft_continuation_seconds
    else:
        continuation_ok = True
    if speeds:
        speed_ok = speed_p5 is not None and speed_p5 >= args.slo_tokens_per_second
    else:
        speed_ok = errors == requests
    error_ok = requests > 0 and error_rate <= args.slo_error_rate
    passed = bool(entry_ok and continuation_ok and speed_ok and error_ok)
    error_samples = []
    for record in records:
        if record.get("error") is None:
            continue
        error_samples.append(
            {
                "status_code": record.get("status_code"),
                "headers": record.get("response_headers"),
                "error": record.get("error"),
            }
        )
        if len(error_samples) >= 5:
            break
    return {
        "sessions": records[0]["step_sessions"] if records else None,
        "requests": requests,
        "errors": errors,
        "error_rate": error_rate,
        "http_codes": http_codes,
        "ttft_entry_seconds": describe(entry_ttft),
        "ttft_continuation_seconds": describe(continuation_ttft),
        "generation_tokens_per_second": {
            "count": len(speeds),
            "p5": speed_p5,
            "p50": percentile(speeds, 0.50),
        },
        "error_samples": error_samples,
        "sla": {
            "ttft_entry_p95_seconds": entry_p95,
            "ttft_entry_limit_seconds": args.slo_ttft_entry_seconds,
            "ttft_entry_ok": entry_ok,
            "ttft_continuation_p95_seconds": continuation_p95,
            "ttft_continuation_limit_seconds": args.slo_ttft_continuation_seconds,
            "ttft_continuation_ok": continuation_ok,
            "generation_p5_tokens_per_second": speed_p5,
            "generation_p5_limit": args.slo_tokens_per_second,
            "generation_p5_ok": speed_ok,
            "error_rate": error_rate,
            "error_rate_limit": args.slo_error_rate,
            "error_rate_ok": error_ok,
            "pass": passed,
        },
    }


def format_wave_verdict(label: str, block: dict[str, Any]) -> str:
    """Однострочный вердикт волны для лога."""
    sla = block["sla"]
    return (
        f"{label} sessions={block['sessions']} requests={block['requests']} "
        f"errors={block['errors']} codes={block['http_codes']} "
        f"entry_ttft_p95={sla['ttft_entry_p95_seconds']} "
        f"cont_ttft_p95={sla['ttft_continuation_p95_seconds']} "
        f"tok_s_p5={sla['generation_p5_tokens_per_second']} "
        f"error_rate={sla['error_rate']} "
        f"verdict={'PASS' if sla['pass'] else 'FAIL'}"
    )


def install_stop_handlers(stop: asyncio.Event) -> None:
    """SIGINT/SIGTERM останавливают волну."""
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, stop.set)
        except NotImplementedError:
            signal.signal(signal_name, lambda *_: stop.set())


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Главный прогон выбранного сценария."""
    apply_scenario_defaults(args)
    if args.sessions < 1 or args.wave_duration_seconds <= 0 or args.target_context_tokens < 1:
        raise ValueError("sessions, duration и target-context-tokens должны быть положительными")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Env-переменная {args.api_key_env} не задана")
    system_template, tools = load_templates(args.templates_dir)
    files = corpus_files(args.corpus_root)
    for file_path in files:
        cached_file_text(file_path)
    stop = asyncio.Event()
    install_stop_handlers(stop)
    all_records: list[dict[str, Any]] = []
    write_lock = asyncio.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "scenario": args.scenario,
        "model": args.model,
        "base_url": args.base_url,
        "target_context_tokens": args.target_context_tokens,
        "max_tokens": args.max_tokens,
        "sessions_requested": args.sessions,
    }
    with args.output.open("w", encoding="utf-8") as output_stream:
        if args.scenario == "steps":
            steps = step_session_counts(args.sessions, args.steps)
            summary["steps"] = []
            first_violation = None
            for size in steps:
                if stop.is_set():
                    break
                LOGGER.info("ступень sessions=%s duration=%ss", size, args.step_duration_seconds)
                wave_records: list[dict[str, Any]] = []
                await run_wave(
                    args,
                    size,
                    args.step_duration_seconds,
                    api_key,
                    system_template,
                    tools,
                    files,
                    output_stream,
                    wave_records,
                    write_lock,
                    stop,
                )
                all_records.extend(wave_records)
                block = sla_block(wave_records, args)
                summary["steps"].append(block)
                LOGGER.info("%s", format_wave_verdict(f"step n={size}", block))
                if first_violation is None and not block["sla"]["pass"]:
                    first_violation = size
                if size != steps[-1] and args.cooldown_seconds > 0 and not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=args.cooldown_seconds)
                    except asyncio.TimeoutError:
                        pass
            summary["first_violation_sessions"] = first_violation
            summary["verdict"] = {
                "pass": first_violation is None,
                "first_violation_sessions": first_violation,
            }
        else:
            await run_wave(
                args,
                args.sessions,
                args.wave_duration_seconds,
                api_key,
                system_template,
                tools,
                files,
                output_stream,
                all_records,
                write_lock,
                stop,
            )
            block = sla_block(all_records, args)
            summary["run"] = block
            summary["verdict"] = {"pass": block["sla"]["pass"]}
            LOGGER.info("%s", format_wave_verdict(args.scenario, block))
    return summary


def self_test() -> None:
    """Проверяет форму Kilo-запроса, уникальность сессий и неизменный префикс."""
    import tempfile

    templates_dir = DEFAULT_TEMPLATES_DIR
    system_template, tools = load_templates(templates_dir)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "alpha.py").write_text("def alpha():\n    return 1\n" * 400, encoding="utf-8")
        (root / "beta.py").write_text("def beta():\n    return 2\n" * 400, encoding="utf-8")
        (root / "gamma.py").write_text("def gamma():\n    return 3\n" * 400, encoding="utf-8")
        files = corpus_files(root)
        first = create_session(1, "test-model", tools, system_template, str(root), DEFAULT_MAX_TOKENS)
        second = create_session(2, "test-model", tools, system_template, str(root), DEFAULT_MAX_TOKENS)
        assert first.session_id != second.session_id
        assert first.affinity != second.affinity
        assert first.affinity.startswith("ses_")
        assert second.affinity.startswith("ses_")
        prepare_session(first, 25_000, 4_000, files)
        prepare_session(second, 25_000, 4_000, files)
        assert any(message["role"] == "assistant" for message in first.messages)
        assert any(message["role"] == "assistant" for message in second.messages)
        payload_one = request_payload(first)
        payload_two = request_payload(second)
        for payload in (payload_one, payload_two):
            assert "temperature" not in payload
            assert payload["max_tokens"] == DEFAULT_MAX_TOKENS
            assert payload["top_p"] == DEFAULT_TOP_P
            assert payload["stream"] is True
            assert payload["stream_options"] == {"include_usage": True}
            assert payload["tool_choice"] == "auto"
            names = tuple(tool["function"]["name"] for tool in payload["tools"])
            assert names == EXPECTED_TOOL_NAMES
            assert payload["messages"][0]["role"] == "system"
            assert isinstance(payload["messages"][0]["content"], str)
            assert first.session_id in payload["messages"][0]["content"] or payload is payload_two
            user = payload["messages"][1]
            assert user["role"] == "user"
            assert isinstance(user["content"], list)
            assert all(part["type"] == "text" and "text" in part for part in user["content"])
            assistant = next(message for message in payload["messages"] if message["role"] == "assistant")
            assert assistant["content"] == KILO_ASSISTANT_CONTENT
            assert isinstance(assistant.get("reasoning_content"), str)
            assert assistant["tool_calls"][0]["type"] == "function"
            assert assistant["tool_calls"][0]["function"]["name"] == "read"
            tool_message = next(message for message in payload["messages"] if message["role"] == "tool")
            assert "def " in tool_message["content"]
        assert first.session_id in payload_one["messages"][0]["content"]
        assert second.session_id in payload_two["messages"][0]["content"]
        assert first.session_id not in payload_two["messages"][0]["content"]
        headers_a = session_headers(first.affinity, "secret")
        headers_b = session_headers(first.affinity, "secret")
        headers_c = session_headers(second.affinity, "secret")
        assert headers_a["x-session-affinity"] == headers_b["x-session-affinity"] == first.affinity
        assert headers_c["x-session-affinity"] == second.affinity
        assert headers_a["x-session-affinity"] != headers_c["x-session-affinity"]
        assert headers_a["User-Agent"] == KILO_USER_AGENT
        assert headers_a["http-referer"] == KILO_REFERER
        assert headers_a["x-title"] == KILO_TITLE
        prefix_len = len(first.messages)
        prefix_one = canonical_json(first.messages)
        append_tool_cycle(first, 200, turn_prompt(first, 1), files)
        assert canonical_json(first.messages[:prefix_len]) == prefix_one
        assert len(first.messages) > prefix_len
        first_tool = next(message["content"] for message in payload_one["messages"] if message["role"] == "tool")
        second_tool = next(message["content"] for message in payload_two["messages"] if message["role"] == "tool")
        assert first_tool != second_tool

    tool_randomizer = random.Random(1234)
    samples = [sample_tool_result_tokens(tool_randomizer) for _ in range(20_000)]
    below_100 = sum(value < 100 for value in samples) / len(samples)
    assert 0.50 <= below_100 <= 0.58
    pacing = random.Random(42)
    pauses = [next_pause_seconds(turn, DEFAULT_BURST_TURNS, pacing) for turn in range(1, 8001)]
    burst_gaps = [pause for turn, pause in zip(range(1, 8001), pauses) if turn % DEFAULT_BURST_TURNS != 0]
    human_pauses = [pause for turn, pause in zip(range(1, 8001), pauses) if turn % DEFAULT_BURST_TURNS == 0]
    assert max(burst_gaps) <= max(AGENT_THINK_SECONDS)
    human_median = percentile(human_pauses, 0.50)
    assert human_median is not None and 90.0 <= human_median <= 190.0

    records = [
        {
            "step_sessions": 2,
            "is_entry": True,
            "ttft_seconds": 10.0,
            "generation_tokens_per_second": 80.0,
            "error": None,
            "empty_completion": False,
            "status_code": 200,
        },
        {
            "step_sessions": 2,
            "is_entry": False,
            "ttft_seconds": 8.0,
            "generation_tokens_per_second": 70.0,
            "error": None,
            "empty_completion": False,
            "status_code": 200,
        },
    ]
    fake_args = argparse.Namespace(
        slo_ttft_entry_seconds=120.0,
        slo_ttft_continuation_seconds=30.0,
        slo_tokens_per_second=50.0,
        slo_error_rate=0.01,
    )
    assert sla_block(records, fake_args)["sla"]["pass"] is True
    LOGGER.info("self-test OK")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if "--self-test" in sys.argv:
        self_test()
        return
    args = parse_args()
    summary = asyncio.run(run(args))
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Raw JSONL: %s", args.output)
    LOGGER.info("Summary JSON: %s", summary_path)
    LOGGER.info("%s", json.dumps(summary, ensure_ascii=False))
    if summary.get("verdict", {}).get("pass") is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
