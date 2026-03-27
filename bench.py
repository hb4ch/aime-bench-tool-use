import os
import re
import ast
import json
import sys
import time
import queue
import argparse
import threading
import subprocess
import warnings
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import httpx
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
# from wrapt_timeout_decorator import timeout

# ============================================================
# Optional pretty logging (rich). Falls back to plain print.
# ============================================================

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None


def log_plain(msg: str) -> None:
    print(msg, flush=True)


def log_info(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[bold cyan][INFO][/bold cyan] {msg}")
    else:
        log_plain(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[bold yellow][WARN][/bold yellow] {msg}")
    else:
        log_plain(f"[WARN] {msg}")


def log_error(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[bold red][ERROR][/bold red] {msg}")
    else:
        log_plain(f"[ERROR] {msg}")


def log_success(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[bold green][OK][/bold green] {msg}")
    else:
        log_plain(f"[OK] {msg}")


def show_banner(title: str, subtitle: Optional[str] = None) -> None:
    if RICH_AVAILABLE:
        text = f"[bold white]{title}[/bold white]"
        if subtitle:
            text += f"\n[dim]{subtitle}[/dim]"
        console.print(Panel(text, border_style="bright_blue", expand=True))
    else:
        log_plain("=" * 80)
        log_plain(title)
        if subtitle:
            log_plain(subtitle)
        log_plain("=" * 80)


def show_config_table(config: Dict[str, Any]) -> None:
    if RICH_AVAILABLE:
        table = Table(title="Run Configuration", box=box.ROUNDED, expand=True)
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="white")
        for k, v in config.items():
            table.add_row(str(k), str(v))
        console.print(table)
    else:
        log_plain("Run Configuration:")
        for k, v in config.items():
            log_plain(f"  - {k}: {v}")


def show_result_panel(title: str, body: str, style: str = "green") -> None:
    if RICH_AVAILABLE:
        console.print(Panel(body, title=title, border_style=style, expand=True))
    else:
        log_plain(f"[{title}] {body}")


warnings.filterwarnings("ignore", category=SyntaxWarning)

# ============================================================
# Explicit configurables
# ============================================================

API_BASE_URL = "http://7.242.108.148:8000/v1"
API_KEY = "REPLACE_ME"
MODEL_NAME = "gpt-oss"

INPUT_DATASET_DIR = "./input/aime25"
OUTPUT_BASE_DIR = "./output"

NO_PROXY_HOST = "7.242.108.148"
DEFAULT_THREAD_NUM = 8
DEFAULT_HTTP_TIMEOUT_SECONDS = 900

MAX_TOOL_ROUNDS = 25
MAX_TOTAL_TURNS = MAX_TOOL_ROUNDS + 1
DEFAULT_TEMPERATURE = 0.1
DEFAULT_REASONING_EFFORT = "medium"
CODE_EXEC_TIMEOUT_SECONDS = 60
MAX_CONSECUTIVE_TOOL_FAILURES = 2
STAGNATION_WINDOW = 3
DEFAULT_WORKER_STALL_TIMEOUT_SECONDS = 120
JOIN_POLL_INTERVAL_SECONDS = 1.0

os.environ["no_proxy"] = NO_PROXY_HOST

# ============================================================
# Regex / parsing helpers
# ============================================================

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
TRAILING_INT_RE = re.compile(r"(-?\d+)\s*$")


def safe_parse_python_like_object(text: str) -> Any:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                try:
                    return ast.literal_eval(obj)
                except Exception:
                    return obj
        return obj
    except Exception:
        pass

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                try:
                    return ast.literal_eval(obj)
                except Exception:
                    return obj
        return obj
    except Exception:
        raise


def normalize_code_string(code: str) -> str:
    """
    Normalize code text safely.

    Cases:
    1. Already real multiline string -> leave it alone
    2. Double-escaped string containing literal '\\n' but no real newlines -> unescape once
    """
    if not isinstance(code, str):
        return code

    if "\n" in code:
        return code

    if "\\n" in code or "\\t" in code or '\\"' in code or "\\'" in code:
        try:
            return code.encode("utf-8").decode("unicode_escape")
        except Exception:
            return code

    return code


def format_code_with_line_numbers(code: str) -> str:
    return "\n".join(f"{i + 1:03d}: {line}" for i, line in enumerate(code.splitlines()))


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_answer_string(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    s = s.replace(",", "")
    return s.strip()


def extract_final_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    boxed_matches = BOXED_RE.findall(text)
    if boxed_matches:
        boxed_text = boxed_matches[-1].strip()
        m = TRAILING_INT_RE.search(boxed_text.replace(",", ""))
        if m:
            return normalize_answer_string(m.group(1).strip())
        return normalize_answer_string(boxed_text)

    m = TRAILING_INT_RE.search(text.strip())
    if m:
        return normalize_answer_string(m.group(1).strip())

    return None


def has_final_answer(text: Optional[str]) -> bool:
    return extract_final_answer(text) is not None


def extract_code_from_legacy_content(content: str) -> str:
    start_marker = "<|message|>"
    end_marker = "<|call|>"

    start = content.find(start_marker)
    end = content.find(end_marker)

    if start < 0 or end < 0 or end <= start:
        raise ValueError("Legacy tool-call markers not found in content.")

    payload = content[start + len(start_marker): end].strip()
    parsed = safe_parse_python_like_object(payload)
    if not isinstance(parsed, dict) or "code" not in parsed:
        raise ValueError("Legacy content parsed, but no 'code' field found.")

    return str(parsed["code"])


def _extract_code_from_mapping(parsed_args: Any, tag: str = "") -> Optional[str]:
    if not isinstance(parsed_args, dict):
        return None

    code = parsed_args.get("code")
    if isinstance(code, str) and code.strip():
        normalized = normalize_code_string(code)
        log_info(f"{tag} | extracted_code_repr={code!r}")
        log_info(f"{tag} | normalized_code_repr={normalized!r}")
        return normalized

    for key in ("python", "input", "arguments"):
        val = parsed_args.get(key)
        if isinstance(val, str) and val.strip():
            normalized = normalize_code_string(val)
            log_info(f"{tag} | extracted_code_from_{key}_repr={val!r}")
            log_info(f"{tag} | normalized_code_from_{key}_repr={normalized!r}")
            return normalized

    return None


def try_extract_raw_tool_code(raw_args: Any, tag: str = "") -> Optional[str]:
    if not isinstance(raw_args, str):
        return None

    normalized = normalize_code_string(raw_args).strip()
    if not normalized:
        return None

    try:
        ast.parse(normalized)
    except SyntaxError as e:
        log_warn(f"{tag} | raw_code_candidate_invalid_python | {type(e).__name__}: {e}")
        return None

    log_info(f"{tag} | raw_code_candidate_accepted")
    return normalized


def looks_like_structured_content(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("<|message|>")


def extract_balanced_json_object_prefix(raw_args: str) -> Optional[str]:
    stripped = raw_args.strip()
    if not stripped.startswith("{"):
        return None

    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(stripped):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[: idx + 1]
            if depth < 0:
                return None

    return None


def try_repair_structured_tool_args(raw_args: Any, tag: str = "") -> Optional[str]:
    if not isinstance(raw_args, str):
        return None

    stripped = raw_args.strip()
    if not stripped or not stripped.startswith("{"):
        return None

    candidates: List[tuple[str, str]] = []
    seen: Set[str] = set()

    def add_candidate(candidate: str, strategy: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append((candidate, strategy))

    prefix = extract_balanced_json_object_prefix(stripped)
    if prefix and prefix != stripped:
        add_candidate(prefix, "trim_trailing_after_balanced_object")

    extra_closing = stripped.count("}") - stripped.count("{")
    trimmed = stripped
    while extra_closing > 0 and trimmed.endswith("}"):
        trimmed = trimmed[:-1].rstrip()
        extra_closing -= 1
        add_candidate(trimmed, "trim_unmatched_trailing_brace")

    for candidate, strategy in candidates:
        try:
            parsed_args = safe_parse_python_like_object(candidate)
        except Exception:
            continue

        extracted = _extract_code_from_mapping(parsed_args, tag=tag)
        if extracted is not None:
            log_info(f"{tag} | repaired_tool_arguments_strategy={strategy}")
            return extracted

    return None


def try_extract_code_request(
    message: Any,
    tag: str = "",
    allow_raw_tool_code_args: bool = False,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if diagnostics is not None:
        diagnostics["repaired_tool_call"] = False

    tool_calls = getattr(message, "tool_calls", None) or []
    content = getattr(message, "content", None)

    if tool_calls:
        raw_args = None
        try:
            raw_args = tool_calls[0].function.arguments
            log_info(f"{tag} | raw_tool_arguments_repr={raw_args!r}")

            parsed_args = safe_parse_python_like_object(raw_args)
            log_info(f"{tag} | parsed_tool_arguments_type={type(parsed_args).__name__}")
            extracted = _extract_code_from_mapping(parsed_args, tag=tag)
            if extracted is not None:
                return extracted
        except Exception as e:
            log_warn(f"{tag} | structured_args_parse_failed | {type(e).__name__}: {e}")

        repaired = try_repair_structured_tool_args(raw_args, tag=tag)
        if repaired is not None:
            if diagnostics is not None:
                diagnostics["repaired_tool_call"] = True
            return repaired

        if allow_raw_tool_code_args:
            extracted = try_extract_raw_tool_code(raw_args, tag=tag)
            if extracted is not None:
                return extracted

    if not content:
        return None

    text = content.strip()
    log_info(f"{tag} | content_repr={text!r}")

    if "<|message|>" in text and "<|call|>" in text:
        try:
            code = extract_code_from_legacy_content(text)
            normalized = normalize_code_string(code)
            log_info(f"{tag} | legacy_extracted_code_repr={code!r}")
            log_info(f"{tag} | legacy_normalized_code_repr={normalized!r}")
            return normalized
        except Exception as e:
            log_warn(f"{tag} | legacy_content_parse_failed | {type(e).__name__}: {e}")

    if looks_like_structured_content(text):
        try:
            parsed = safe_parse_python_like_object(text)
            log_info(f"{tag} | parsed_content_type={type(parsed).__name__}")
            extracted = _extract_code_from_mapping(parsed, tag=tag)
            if extracted is not None:
                return extracted
        except Exception as e:
            log_warn(f"{tag} | content_parse_failed | {type(e).__name__}: {e}")

    return None


def detect_obviously_broken_tool_code(code: str) -> Optional[str]:
    stripped = code.strip()
    if not stripped:
        return "empty code"

    placeholder_patterns = (
        r"=\s*$",
        r"\(\s*,",
        r",\s*,",
        r"=\s*$",
    )
    for pattern in placeholder_patterns:
        if re.search(pattern, stripped, flags=re.MULTILINE):
            return "contains an unfinished placeholder"

    for line in stripped.splitlines():
        if re.search(r"=\s*$", line):
            return "contains an assignment with no expression"

    try:
        ast.parse(stripped)
    except SyntaxError as e:
        return f"invalid Python syntax: {e}"

    return None


def fingerprint_tool_payload(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    normalized = normalize_whitespace(text)
    if not normalized:
        return None
    return normalized[:300]


def build_tool_retry_prompt(reason: str) -> str:
    return (
        "Your previous tool code was not executed because it was obviously malformed "
        f"({reason}). Reissue a corrected python_exec call with valid JSON arguments "
        'like {"code": "print(1+1)"} and complete Python code only.'
    )


def build_finalization_prompt_from_tool_result(tool_output: Optional[str] = None) -> str:
    if tool_output:
        return (
            "You are out of tool budget or should stop using tools.\n"
            "Use the latest Python result below to finish the problem.\n"
            "Do not call any more tools.\n"
            "Give exactly one final boxed answer like \\boxed{123}.\n"
            f"Latest Python result:\n{tool_output}"
        )
    return build_finalization_prompt()


def should_force_finalization_early(
    recent_tool_events: List[Dict[str, Optional[str]]],
    consecutive_tool_failures: int,
    remaining_turns_after_current: int,
) -> bool:
    if remaining_turns_after_current <= 1 and recent_tool_events:
        return True
    if consecutive_tool_failures >= MAX_CONSECUTIVE_TOOL_FAILURES:
        return True
    if len(recent_tool_events) < STAGNATION_WINDOW:
        return False

    window = recent_tool_events[-STAGNATION_WINDOW:]
    if all(event.get("status") != "ok" for event in window):
        return True

    code_fingerprints = [event.get("code_fingerprint") for event in window if event.get("code_fingerprint")]
    if len(code_fingerprints) >= 2 and len(set(code_fingerprints)) == 1:
        return True

    output_fingerprints = [
        event.get("output_fingerprint") for event in window
        if event.get("status") == "ok" and event.get("output_fingerprint")
    ]
    if len(output_fingerprints) >= 2 and len(set(output_fingerprints)) == 1:
        return True

    return False


def normalize_code_to_print_last_expression(code_content: str) -> str:
    lines = code_content.strip().split("\n")

    while lines:
        tail = lines[-1].strip()
        if tail == "" or tail == "print()":
            lines.pop()
        else:
            break

    if not lines:
        return "print()"

    last_line = lines[-1]
    stripped = last_line.strip()

    top_level = not last_line.startswith((" ", "\t"))
    statement_prefixes = (
        "print(",
        "for ",
        "while ",
        "if ",
        "elif ",
        "else:",
        "try:",
        "except ",
        "finally:",
        "with ",
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "raise ",
        "assert ",
        "del ",
        "pass",
        "break",
        "continue",
        "@",
    )

    if top_level and stripped and not stripped.startswith("#") and not stripped.startswith(statement_prefixes):
        assignment_like = (
            "=" in stripped and "==" not in stripped and ":=" not in stripped
            and "<=" not in stripped and ">=" not in stripped and "!=" not in stripped
        )
        if not assignment_like:
            if "#" in stripped:
                stripped = stripped[: stripped.find("#")].strip()
            lines[-1] = f"print({stripped})"

    return "\n".join(lines)


def indent_code_block(code_content: str) -> str:
    return "\n".join(" " * 4 + line for line in code_content.split("\n"))


def build_assistant_message_dict(message: Any) -> Dict[str, Any]:
    assistant_msg: Dict[str, Any] = {
        "role": "assistant",
        "content": message.content if getattr(message, "content", None) is not None else "",
    }

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        assistant_msg["tool_calls"] = []
        for tc in tool_calls:
            assistant_msg["tool_calls"].append(
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )

    return assistant_msg


def build_tool_message(tool_call_id: str, func_name: str, content: str) -> Dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": func_name,
        "content": content,
    }


def build_finalization_prompt() -> str:
    return (
        "Do not call python_exec again. Based on the work so far, provide the final answer now. "
        "Your reply must contain exactly one boxed value like \\boxed{123}, and you must not box intermediate quantities."
    )


# ============================================================
# Code execution / worker monitoring
# ============================================================

@dataclass
class ToolExecutionResult:
    status: str
    stdout: str
    error: Optional[str] = None


@dataclass
class WorkerState:
    sample_id: Optional[str]
    phase: str
    updated_at: float
    last_reported_at: float = 0.0


def sanitize_path_component(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return sanitized or "unknown"


def build_tool_runner_script(prefix: str, prior_snippets: List[str], current_snippet: str) -> str:
    sections: List[str] = [
        "import sys",
        "from io import StringIO",
        "from contextlib import redirect_stdout",
        prefix.rstrip("\n"),
        "",
        "_bench_replay_stdout = StringIO()",
    ]

    for snippet in prior_snippets:
        if not snippet.strip():
            continue
        sections.append("with redirect_stdout(_bench_replay_stdout):")
        sections.append(indent_code_block(snippet))
        sections.append("")

    sections.extend(
        [
            "_bench_current_stdout = StringIO()",
            "with redirect_stdout(_bench_current_stdout):",
            indent_code_block(current_snippet),
            "sys.stdout.write(_bench_current_stdout.getvalue())",
        ]
    )
    return "\n".join(section for section in sections if section is not None)


def execute_python_code_subprocess(
    prefix: str,
    prior_snippets: List[str],
    current_snippet: str,
    scratch_dir: Path,
    timeout_seconds: int,
) -> ToolExecutionResult:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runner_code = build_tool_runner_script(prefix, prior_snippets, current_snippet)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            cwd=str(scratch_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolExecutionResult(
            status="timeout",
            stdout="",
            error=f"Tool execution timed out after {timeout_seconds} seconds.",
        )
    except Exception as e:
        return ToolExecutionResult(
            status="error",
            stdout="",
            error=f"{type(e).__name__}: {e}",
        )

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or f"subprocess exited with code {proc.returncode}").strip()
        return ToolExecutionResult(
            status="error",
            stdout=proc.stdout,
            error=err,
        )

    return ToolExecutionResult(status="ok", stdout=proc.stdout)


def update_worker_state(
    worker_states: Dict[str, WorkerState],
    worker_state_lock: threading.Lock,
    worker_name: str,
    phase: str,
    sample_id: Optional[str] = None,
) -> None:
    now = time.time()
    with worker_state_lock:
        existing = worker_states.get(worker_name)
        last_reported_at = existing.last_reported_at if existing else 0.0
        if existing and existing.phase != phase:
            last_reported_at = 0.0
        worker_states[worker_name] = WorkerState(
            sample_id=sample_id,
            phase=phase,
            updated_at=now,
            last_reported_at=last_reported_at,
        )


def collect_stale_workers(
    worker_states: Dict[str, WorkerState],
    worker_state_lock: threading.Lock,
    stall_timeout_seconds: int,
) -> List[WorkerState]:
    now = time.time()
    stale_workers: List[WorkerState] = []

    with worker_state_lock:
        for worker_name, state in worker_states.items():
            if state.phase in {"finished", "idle"}:
                continue
            age = now - state.updated_at
            if age < stall_timeout_seconds:
                continue
            if state.last_reported_at and now - state.last_reported_at < stall_timeout_seconds:
                continue
            state.last_reported_at = now
            stale_workers.append(
                WorkerState(
                    sample_id=f"{worker_name}:{state.sample_id or 'unknown'}",
                    phase=state.phase,
                    updated_at=state.updated_at,
                    last_reported_at=state.last_reported_at,
                )
            )

    return stale_workers


def join_threads_with_monitoring(
    threads: List[threading.Thread],
    worker_states: Dict[str, WorkerState],
    worker_state_lock: threading.Lock,
    stall_timeout_seconds: int,
) -> None:
    per_thread_join_timeout = JOIN_POLL_INTERVAL_SECONDS / max(len(threads), 1)
    while True:
        any_alive = False
        for thread in threads:
            thread.join(timeout=per_thread_join_timeout)
            if thread.is_alive():
                any_alive = True

        for state in collect_stale_workers(worker_states, worker_state_lock, stall_timeout_seconds):
            worker_name, _, sample_id = state.sample_id.partition(":")
            sample_display = sample_id or "unknown"
            age = time.time() - state.updated_at
            log_warn(
                f"{worker_name} | sample={sample_display} | phase={state.phase} "
                f"| stalled_for={age:.1f}s"
            )

        if not any_alive:
            return


# ============================================================
# Scoring helpers
# ============================================================

def judge_one_sample(pred_answer: Optional[str], gold_answer: Optional[str]) -> Dict[str, Any]:
    pred = normalize_answer_string(pred_answer)
    gold = normalize_answer_string(gold_answer)

    if gold is None:
        return {
            "gold_answer": gold,
            "pred_answer": pred,
            "is_scored": False,
            "is_correct": None,
        }

    if pred is None:
        return {
            "gold_answer": gold,
            "pred_answer": pred,
            "is_scored": True,
            "is_correct": False,
        }

    return {
        "gold_answer": gold,
        "pred_answer": pred,
        "is_scored": True,
        "is_correct": pred == gold,
    }


def init_score_counter() -> Dict[str, int]:
    return {
        "total": 0,
        "scored_total": 0,
        "answered": 0,
        "correct": 0,
        "wrong": 0,
        "no_final_answer": 0,
        "tool_timeout": 0,
        "request_error": 0,
        "response_error": 0,
        "other_end": 0,
        "function_call_attempts": 0,
        "function_call_parse_success": 0,
        "function_call_parse_fail": 0,
        "function_call_success": 0,
        "function_call_timeout": 0,
        "function_call_error": 0,
        "repaired_tool_calls": 0,
        "forced_finalization_turns": 0,
    }


def update_score_counter(counter: Dict[str, int], record: Dict[str, Any]) -> None:
    counter["total"] += 1

    end_type = record.get("end_type")
    if record.get("pred_answer") is not None:
        counter["answered"] += 1

    if end_type == "no_final_answer":
        counter["no_final_answer"] += 1
    elif end_type == "tool_timeout":
        counter["tool_timeout"] += 1
    elif end_type == "request_error":
        counter["request_error"] += 1
    elif end_type == "response_error":
        counter["response_error"] += 1
    elif end_type not in {"answered", None}:
        counter["other_end"] += 1

    counter["function_call_attempts"] += record.get("function_call_attempts", 0)
    counter["function_call_parse_success"] += record.get("function_call_parse_success", 0)
    counter["function_call_parse_fail"] += record.get("function_call_parse_fail", 0)
    counter["function_call_success"] += record.get("function_call_success", 0)
    counter["function_call_timeout"] += record.get("function_call_timeout", 0)
    counter["function_call_error"] += record.get("function_call_error", 0)
    counter["repaired_tool_calls"] += record.get("repaired_tool_calls", 0)
    counter["forced_finalization_turns"] += record.get("forced_finalization_turns", 0)

    if record.get("is_scored"):
        counter["scored_total"] += 1
        if record.get("is_correct") is True:
            counter["correct"] += 1
        elif record.get("is_correct") is False:
            counter["wrong"] += 1


def accuracy_str(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{num}/{den} = {100.0 * num / den:.2f}%"


def parse_csv_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_openrouter_provider_preferences(
    provider_order: Optional[str] = None,
    provider_quantizations: Optional[str] = None,
    allow_fallbacks: bool = True,
) -> Optional[Dict[str, Any]]:
    preferences: Dict[str, Any] = {}
    order = parse_csv_arg(provider_order)
    quantizations = parse_csv_arg(provider_quantizations)

    if order:
        preferences["order"] = order
    if quantizations:
        preferences["quantizations"] = quantizations
    if not allow_fallbacks:
        preferences["allow_fallbacks"] = False

    return preferences or None


def extract_function_call_stats(response_items: Any) -> Dict[str, int]:
    stats = {
        "function_call_attempts": 0,
        "function_call_parse_success": 0,
        "function_call_parse_fail": 0,
        "function_call_success": 0,
        "function_call_timeout": 0,
        "function_call_error": 0,
    }
    if not isinstance(response_items, list):
        return stats

    for item in response_items:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "assistant" and item.get("finish_reason") == "tool_calls":
            stats["function_call_attempts"] += 1
        if item.get("role") != "tool":
            continue

        stats["function_call_parse_success"] += 1
        tool_status = item.get("tool_status")
        if tool_status == "ok":
            stats["function_call_success"] += 1
        elif tool_status == "timeout":
            stats["function_call_timeout"] += 1
        elif tool_status == "error":
            stats["function_call_error"] += 1

    if stats["function_call_attempts"] < stats["function_call_parse_success"]:
        stats["function_call_attempts"] = stats["function_call_parse_success"]
    stats["function_call_parse_fail"] = (
        stats["function_call_attempts"] - stats["function_call_parse_success"]
    )
    return stats


def summarize_jsonl_scores(jsonl_path: str) -> Dict[str, Any]:
    counter = init_score_counter()
    rows: List[Dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            row = {
                "id": obj.get("id"),
                "pred_answer": obj.get("pred_answer"),
                "gold_answer": obj.get("gold_answer"),
                "is_scored": obj.get("is_scored"),
                "is_correct": obj.get("is_correct"),
                "end_type": obj.get("end_type"),
                "repaired_tool_calls": obj.get("repaired_tool_calls", 0),
                "forced_finalization_turns": obj.get("forced_finalization_turns", 0),
            }
            row.update(extract_function_call_stats(obj.get("response")))
            rows.append(row)
            update_score_counter(counter, row)

    return {
        "counts": counter,
        "accuracy_all": accuracy_str(counter["correct"], counter["total"]),
        "accuracy_scored": accuracy_str(counter["correct"], counter["scored_total"]),
        "accuracy_answered": accuracy_str(counter["correct"], counter["answered"]),
        "function_call_success_rate": accuracy_str(
            counter["function_call_success"],
            counter["function_call_attempts"],
        ),
        "function_call_parse_success_rate": accuracy_str(
            counter["function_call_parse_success"],
            counter["function_call_attempts"],
        ),
        "rows": rows,
    }


def show_score_table(title: str, score_summary: Dict[str, Any]) -> None:
    counts = score_summary["counts"]

    if RICH_AVAILABLE:
        table = Table(title=title, box=box.ROUNDED, expand=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="white")
        table.add_row("total", str(counts["total"]))
        table.add_row("scored_total", str(counts["scored_total"]))
        table.add_row("answered", str(counts["answered"]))
        table.add_row("correct", str(counts["correct"]))
        table.add_row("wrong", str(counts["wrong"]))
        table.add_row("no_final_answer", str(counts["no_final_answer"]))
        table.add_row("tool_timeout", str(counts["tool_timeout"]))
        table.add_row("request_error", str(counts["request_error"]))
        table.add_row("response_error", str(counts["response_error"]))
        table.add_row("other_end", str(counts["other_end"]))
        table.add_row("function_call_attempts", str(counts["function_call_attempts"]))
        table.add_row("function_call_parse_success", str(counts["function_call_parse_success"]))
        table.add_row("function_call_parse_fail", str(counts["function_call_parse_fail"]))
        table.add_row("function_call_success", str(counts["function_call_success"]))
        table.add_row("function_call_timeout", str(counts["function_call_timeout"]))
        table.add_row("function_call_error", str(counts["function_call_error"]))
        table.add_row("repaired_tool_calls", str(counts["repaired_tool_calls"]))
        table.add_row("forced_finalization_turns", str(counts["forced_finalization_turns"]))
        table.add_row("accuracy_all", score_summary["accuracy_all"])
        table.add_row("accuracy_scored", score_summary["accuracy_scored"])
        table.add_row("accuracy_answered", score_summary["accuracy_answered"])
        table.add_row("function_call_success_rate", score_summary["function_call_success_rate"])
        table.add_row("function_call_parse_success_rate", score_summary["function_call_parse_success_rate"])
        console.print(table)
    else:
        log_plain(f"== {title} ==")
        for k, v in counts.items():
            log_plain(f"{k}: {v}")
        log_plain(f"accuracy_all: {score_summary['accuracy_all']}")
        log_plain(f"accuracy_scored: {score_summary['accuracy_scored']}")
        log_plain(f"accuracy_answered: {score_summary['accuracy_answered']}")
        log_plain(f"function_call_success_rate: {score_summary['function_call_success_rate']}")
        log_plain(f"function_call_parse_success_rate: {score_summary['function_call_parse_success_rate']}")


# ============================================================
# Model client / controller
# ============================================================

class BenchmarkClient:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
        openrouter_provider_preferences: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        if OpenAI is None:
            raise ImportError("openai package is required to construct BenchmarkClient")
        self.model_name = model_name
        self.httpx_client = httpx.Client(verify=False, timeout=http_timeout_seconds)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=self.httpx_client,
        )
        self.openrouter_provider_preferences = openrouter_provider_preferences
        self.verbose = verbose

    def close(self) -> None:
        self.httpx_client.close()

    def _system_prompt(self) -> str:
        return (
            "You are a math problem solver that may use Python code when calculation is needed.\n"
            "Rules:\n"
            "1. Use Python only when it helps with nontrivial algebra, arithmetic, or checking.\n"
            "2. Any Python code you send to the tool must print its result explicitly.\n"
            "3. When you have enough information, stop using tools.\n"
            "4. Your final response must contain exactly one boxed final answer like \\boxed{123}.\n"
            "5. Do not box intermediate quantities.\n"
            "6. Do not keep calling tools once the answer is determined.\n"
            "7. Any tool call must use valid JSON arguments with exactly one field named code.\n"
            "8. Do not include markdown fences, extra keys, or prose in tool arguments.\n"
            "9. Avoid naming the variable 'n' as it may have an effect on the evaluation environment.\n"
        )

    def _tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "python_exec",
                    "description": "Execute Python code and return stdout as a string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. It should print the result explicitly.",
                            },
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def _exec_prefix(self) -> str:
        return (
            "import sys\n"
            "import math\n"
            "import numpy as np\n"
            "import sympy\n"
            "from sympy import *\n"
            "from fractions import Fraction\n"
            "from numpy import mean, std, median\n"
            "from io import StringIO\n"
            "from contextlib import redirect_stdout\n"
            "from datetime import timedelta\n"
            "if hasattr(sys, 'set_int_max_str_digits'):\n"
            "    sys.set_int_max_str_digits(0)\n"
        )

    def _call_model(
        self,
        messages: List[Dict[str, Any]],
        use_tools: bool,
        temperature: float,
        reasoning_effort: str,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
        }
        if use_tools:
            kwargs["tools"] = self._tools()
            kwargs["tool_choice"] = "auto"
        if self.openrouter_provider_preferences:
            kwargs["extra_body"] = {"provider": self.openrouter_provider_preferences}
        return self.client.chat.completions.create(**kwargs)

    def solve_one(
        self,
        user_input: str,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        max_total_turns: int = MAX_TOTAL_TURNS,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        sample_id: Optional[str] = None,
        worker_name: Optional[str] = None,
        sample_scratch_dir: Optional[Path] = None,
        code_exec_timeout_seconds: int = CODE_EXEC_TIMEOUT_SECONDS,
        allow_raw_tool_code_args: bool = False,
        phase_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        tag = f"{worker_name or 'worker'}"
        if sample_id:
            tag += f" | {sample_id}"

        result_list: List[Dict[str, Any]] = [{"role": "user", "content": user_input}]
        msgs: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_input},
        ]

        prefix = self._exec_prefix()
        successful_tool_snippets: List[str] = []
        tool_timed_out = False
        tool_rounds = 0
        total_turns = 0
        effective_max_total_turns = max(max_total_turns, max_tool_rounds + 1)
        last_error: Optional[str] = None
        finalization_prompt_sent = False
        repaired_tool_call_count = 0
        forced_finalization_turns = 0
        consecutive_tool_failures = 0
        recent_tool_events: List[Dict[str, Optional[str]]] = []
        last_tool_output: Optional[str] = None
        if sample_scratch_dir is None:
            sample_scratch_dir = Path.cwd()
        sample_scratch_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            log_info(f"{tag} | start")

        while total_turns < effective_max_total_turns:
            use_tools = (
                tool_rounds < max_tool_rounds
                and total_turns < effective_max_total_turns - 1
                and not should_force_finalization_early(
                    recent_tool_events,
                    consecutive_tool_failures,
                    effective_max_total_turns - total_turns - 1,
                )
            )
            forcing_finalization = not use_tools

            if self.verbose:
                log_info(
                    f"{tag} | turn={total_turns + 1}/{effective_max_total_turns} "
                    f"| tool_rounds={tool_rounds}/{max_tool_rounds} | use_tools={use_tools}"
                )

            if forcing_finalization and not finalization_prompt_sent and msgs:
                msgs.append(
                    {
                        "role": "user",
                        "content": build_finalization_prompt_from_tool_result(last_tool_output),
                    }
                )
                finalization_prompt_sent = True
                forced_finalization_turns += 1
                log_info(f"{tag} | forced_finalization_prompt_sent")

            try:
                if phase_callback:
                    phase_callback("model")
                response = self._call_model(
                    messages=msgs,
                    use_tools=use_tools,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                result_list.append(
                    {
                        "role": "assistant",
                        "reasoning_content": None,
                        "content": None,
                        "request_error": err,
                    }
                )
                result_list.extend(
                    [
                        {"iter_num": total_turns},
                        {"end_type": "request_error"},
                    ]
                )
                log_error(f"{tag} | request_error | {err}")
                return result_list

            if not response or not getattr(response, "choices", None):
                result_list.append({"role": "assistant", "reasoning_content": None, "content": None})
                result_list.extend(
                    [
                        {"iter_num": total_turns},
                        {"end_type": "response_error"},
                    ]
                )
                log_error(f"{tag} | empty response")
                return result_list

            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason
            content = getattr(message, "content", None)
            reasoning_content = getattr(message, "reasoning_content", None)
            tool_calls = getattr(message, "tool_calls", None) or []
            code_request_diagnostics: Dict[str, Any] = {}
            code_request = try_extract_code_request(
                message,
                tag=tag,
                allow_raw_tool_code_args=allow_raw_tool_code_args,
                diagnostics=code_request_diagnostics,
            )
            if code_request_diagnostics.get("repaired_tool_call"):
                repaired_tool_call_count += 1

            result_list.append(
                {
                    "role": "assistant",
                    "reasoning_content": reasoning_content,
                    "content": content,
                    "finish_reason": finish_reason,
                }
            )

            answer = extract_final_answer(content)

            if content:
                preview = content.replace("\n", " ").strip()
                if len(preview) > 180:
                    preview = preview[:180] + "..."
                log_info(f"{tag} | finish_reason={finish_reason} | content={preview!r}")
            else:
                log_info(f"{tag} | finish_reason={finish_reason} | content=None")

            if answer is not None:
                result_list.extend(
                    [
                        {"iter_num": total_turns},
                        {"end_type": "answered"},
                        {"final_answer": answer},
                        {"repaired_tool_call_count": repaired_tool_call_count},
                        {"forced_finalization_turns": forced_finalization_turns},
                    ]
                )
                log_success(f"{tag} | answered | final_answer={answer}")
                return result_list

            if finish_reason == "tool_calls" and not use_tools:
                log_warn(f"{tag} | tool_call_requested_during_finalization")
                msgs.append(build_assistant_message_dict(message))
                msgs.append(
                    {
                        "role": "user",
                        "content": build_finalization_prompt_from_tool_result(last_tool_output),
                    }
                )
                total_turns += 1
                continue

            if finish_reason == "tool_calls" and use_tools and code_request is None:
                log_warn(f"{tag} | finish_reason=tool_calls but no parseable code_request found")
                raw_args = None
                try:
                    if tool_calls:
                        raw_args = tool_calls[0].function.arguments
                except Exception:
                    pass
                log_warn(f"{tag} | raw_args={raw_args!r}")

                msgs.append(build_assistant_message_dict(message))
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response requested a tool call, but the tool arguments could not be parsed. "
                            'Please call python_exec again with valid JSON arguments like {"code": "print(1+1)"}'
                        ),
                    }
                )
                total_turns += 1
                continue

            if code_request is not None and use_tools:
                try:
                    log_info(f"{tag} | code_request_repr={code_request!r}")
                    log_info(f"{tag} | code_request_with_lines=\n{format_code_with_line_numbers(code_request)}")

                    raw_code = textwrap.dedent(code_request).strip("\n")
                    log_info(f"{tag} | raw_code_after_dedent_repr={raw_code!r}")
                    log_info(f"{tag} | raw_code_after_dedent_lines=\n{format_code_with_line_numbers(raw_code)}")

                    code_to_run = normalize_code_to_print_last_expression(raw_code)
                    log_info(f"{tag} | code_after_normalize_repr={code_to_run!r}")
                    log_info(f"{tag} | code_after_normalize_lines=\n{format_code_with_line_numbers(code_to_run)}")

                    broken_code_reason = detect_obviously_broken_tool_code(code_to_run)
                    if broken_code_reason is not None:
                        log_warn(f"{tag} | rejected_broken_tool_code | {broken_code_reason}")
                        msgs.append(build_assistant_message_dict(message))
                        msgs.append(
                            {
                                "role": "user",
                                "content": build_tool_retry_prompt(broken_code_reason),
                            }
                        )
                        total_turns += 1
                        continue

                    code_str = indent_code_block(code_to_run)
                    log_info(f"{tag} | code_after_indent_repr={code_str!r}")
                    log_info(f"{tag} | tool_intent_detected | executing_code_lines=\n{format_code_with_line_numbers(code_str)}")

                    if phase_callback:
                        phase_callback("tool")
                    tool_result = execute_python_code_subprocess(
                        prefix=prefix,
                        prior_snippets=successful_tool_snippets,
                        current_snippet=code_to_run,
                        scratch_dir=sample_scratch_dir,
                        timeout_seconds=code_exec_timeout_seconds,
                    )
                    if tool_result.status == "ok":
                        ret = tool_result.stdout
                        successful_tool_snippets.append(code_to_run)
                    elif tool_result.status == "timeout":
                        tool_timed_out = True
                        last_error = "tool_timeout"
                        ret = f"ToolTimeout: {tool_result.error}"
                        log_warn(f"{tag} | tool_exec_timeout | {tool_result.error}")
                    else:
                        ret = f"ToolError: {tool_result.error}"
                        log_warn(f"{tag} | tool_exec_error | {tool_result.error}")
                except Exception as e:
                    ret = f"{type(e).__name__}: {e}"
                    code_to_run = ""
                    log_warn(f"{tag} | tool_exec_error | {ret}")

                tool_preview = ret.replace("\n", " ").strip()
                if len(tool_preview) > 180:
                    tool_preview = tool_preview[:180] + "..."
                log_info(f"{tag} | tool_output={tool_preview!r}")
                tool_status = "ok"
                if ret.startswith("ToolTimeout:"):
                    tool_status = "timeout"
                elif ret.startswith("ToolError:"):
                    tool_status = "error"
                if tool_status == "ok":
                    consecutive_tool_failures = 0
                    last_tool_output = ret
                else:
                    consecutive_tool_failures += 1
                recent_tool_events.append(
                    {
                        "status": tool_status,
                        "code_fingerprint": fingerprint_tool_payload(code_to_run),
                        "output_fingerprint": fingerprint_tool_payload(ret),
                    }
                )
                if len(recent_tool_events) > STAGNATION_WINDOW:
                    recent_tool_events = recent_tool_events[-STAGNATION_WINDOW:]

                if tool_calls:
                    first_tool_call = tool_calls[0]
                    tool_message = build_tool_message(
                        tool_call_id=first_tool_call.id,
                        func_name=first_tool_call.function.name,
                        content=ret,
                    )
                    msgs.append(build_assistant_message_dict(message))
                    next_total_turns = total_turns + 1
                    next_tool_rounds = tool_rounds + 1
                    force_finalize_after_tool = (
                        next_tool_rounds >= max_tool_rounds
                        or next_total_turns >= effective_max_total_turns - 1
                        or should_force_finalization_early(
                            recent_tool_events,
                            consecutive_tool_failures,
                            effective_max_total_turns - next_total_turns,
                        )
                    )
                    followup_content = (
                        build_finalization_prompt_from_tool_result(ret)
                        if force_finalize_after_tool
                        else (
                            "Python execution result:\n"
                            f"{ret}\n"
                            "Continue solving. Use tools again if needed. "
                            "When done, give the final answer within \\boxed{}."
                        )
                    )
                    if force_finalize_after_tool and not finalization_prompt_sent:
                        finalization_prompt_sent = True
                        forced_finalization_turns += 1
                        log_info(f"{tag} | forced_finalization_prompt_sent")
                    msgs.append(tool_message)
                    msgs.append({"role": "user", "content": followup_content})
                    result_list.append(
                        {
                            "role": "tool",
                            "code_snippet": code_to_run,
                            "tool_status": tool_status,
                            "tool_message": tool_message,
                        }
                    )
                else:
                    msgs.append(
                        {
                            "role": "assistant",
                            "content": content if content is not None else "",
                        }
                    )
                    next_total_turns = total_turns + 1
                    next_tool_rounds = tool_rounds + 1
                    force_finalize_after_tool = (
                        next_tool_rounds >= max_tool_rounds
                        or next_total_turns >= effective_max_total_turns - 1
                    )
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                build_finalization_prompt_from_tool_result(ret)
                                if force_finalize_after_tool
                                else (
                                    "Python execution result:\n"
                                    f"{ret}\n"
                                    "Continue solving. Use tools again if needed. "
                                    "When done, give the final answer within \\boxed{}."
                                )
                            ),
                        }
                    )
                    if force_finalize_after_tool and not finalization_prompt_sent:
                        finalization_prompt_sent = True
                        forced_finalization_turns += 1
                        log_info(f"{tag} | forced_finalization_prompt_sent")
                    result_list.append(
                        {
                            "role": "tool",
                            "code_snippet": code_to_run,
                            "tool_status": tool_status,
                            "tool_message": {
                                "role": "tool",
                                "name": "python_exec",
                                "content": ret,
                            },
                        }
                    )

                tool_rounds += 1
                total_turns += 1
                continue

            last_error = f"no_final_answer_this_turn (finish_reason={finish_reason})"
            log_warn(f"{tag} | {last_error}")
            total_turns += 1
            continue

        result_list.extend(
            [
                {"iter_num": total_turns},
                {"end_type": "tool_timeout" if tool_timed_out else "no_final_answer"},
                {"repaired_tool_call_count": repaired_tool_call_count},
                {"forced_finalization_turns": forced_finalization_turns},
            ]
        )
        if last_error:
            result_list.append({"controller_note": last_error})
        return result_list


# ============================================================
# JSONL helpers
# ============================================================

def merge_query_set(jsonl_dir: str) -> Set[str]:
    query_set: Set[str] = set()

    for root, _, files in os.walk(jsonl_dir):
        for file_name in files:
            if not (file_name.endswith(".jsonl") or file_name.endswith(".json")):
                continue

            file_path = os.path.join(root, file_name)
            with open(file_path, "r", encoding="utf-8") as reader_f:
                for line in reader_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_line = json.loads(line)
                        prompt = json_line.get("prompt")
                        if prompt:
                            query_set.add(prompt)
                    except Exception as e:
                        log_warn(f"failed to parse line in {file_path}: {e}")

    log_success("merge query done")
    return query_set


def merge_jsonls(jsonl_dir: str, save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as writer_f:
        for root, _, files in os.walk(jsonl_dir):
            for file_name in files:
                if not (file_name.endswith(".jsonl") or file_name.endswith(".json")):
                    continue

                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as reader_f:
                    for line in reader_f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json_line = json.loads(line)
                            writer_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                        except Exception as e:
                            log_warn(f"failed to merge line from {file_path}: {e}")

    log_success(f"merged jsonl saved to: {save_path}")


# ============================================================
# Worker
# ============================================================

class BenchmarkWorker(threading.Thread):
    def __init__(
        self,
        thread_id: int,
        task_queue: queue.Queue,
        target_filename_pattern: str,
        client_factory: Callable[[], Any],
        max_tool_rounds: int,
        max_total_turns: int,
        temperature: float,
        reasoning_effort: str,
        scratch_root: Path,
        code_exec_timeout_seconds: int,
        allow_raw_tool_code_args: bool,
        worker_states: Dict[str, WorkerState],
        worker_state_lock: threading.Lock,
    ):
        super().__init__(name=f"thread-{thread_id}")
        self.thread_id = thread_id
        self.worker_name = f"thread-{thread_id}"
        self.filename = target_filename_pattern.replace("*", f"_thread{thread_id}.jsonl")
        self.task_queue = task_queue
        self.client_factory = client_factory
        self.max_tool_rounds = max_tool_rounds
        self.max_total_turns = max_total_turns
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.scratch_root = scratch_root
        self.code_exec_timeout_seconds = code_exec_timeout_seconds
        self.allow_raw_tool_code_args = allow_raw_tool_code_args
        self.worker_states = worker_states
        self.worker_state_lock = worker_state_lock

    def _clean_question(self, question: str) -> str:
        suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        if suffix in question:
            question = question[: question.find(suffix)]
        return question

    def _set_state(self, phase: str, sample_id: Optional[str] = None) -> None:
        update_worker_state(
            worker_states=self.worker_states,
            worker_state_lock=self.worker_state_lock,
            worker_name=self.worker_name,
            phase=phase,
            sample_id=sample_id,
        )

    def run(self) -> None:
        log_info(f"{self.worker_name} | started | output={self.filename}")
        client = None
        self._set_state("idle")

        try:
            client = self.client_factory()
            while True:
                self._set_state("queue")
                try:
                    line = self.task_queue.get_nowait()
                except queue.Empty:
                    break

                try:
                    json_line = json.loads(line)
                except Exception as e:
                    log_warn(f"{self.worker_name} | malformed input line skipped | {e}")
                    continue

                question = self._clean_question(json_line["prompt"])
                sample_id = str(json_line.get("id", "unknown"))
                gold_answer = json_line.get("answer")
                sample_scratch_dir = (
                    self.scratch_root
                    / sanitize_path_component(self.worker_name)
                    / sanitize_path_component(sample_id)
                )

                response_res: Optional[List[Dict[str, Any]]] = None
                quota_tag = False

                for attempt in range(1, 4):
                    log_info(f"{self.worker_name} | sample={sample_id} | attempt={attempt}/3")
                    try:
                        self._set_state("model", sample_id)
                        response_res = client.solve_one(
                            question,
                            max_tool_rounds=self.max_tool_rounds,
                            max_total_turns=self.max_total_turns,
                            temperature=self.temperature,
                            reasoning_effort=self.reasoning_effort,
                            sample_id=sample_id,
                            worker_name=self.worker_name,
                            sample_scratch_dir=sample_scratch_dir,
                            code_exec_timeout_seconds=self.code_exec_timeout_seconds,
                            allow_raw_tool_code_args=self.allow_raw_tool_code_args,
                            phase_callback=lambda phase, sid=sample_id: self._set_state(phase, sid),
                        )
                        if response_res:
                            break
                    except Exception as e:
                        err_text = str(e)
                        log_warn(f"{self.worker_name} | sample={sample_id} | err={err_text}")
                        quota_exception = (
                            "quota" in err_text
                            or "new_api_error" in err_text
                            or "AuthenticationError" in err_text
                        )
                        if quota_exception:
                            quota_tag = True
                            log_warn(f"{self.worker_name} | sample={sample_id} | quota/auth issue, skip")
                            break
                        response_res = None
                        json_line["request_exception"] = True

                if quota_tag:
                    continue

                pred_answer = None
                end_type = None
                repaired_tool_call_count = 0
                forced_finalization_turns = 0
                if isinstance(response_res, list):
                    for item in response_res:
                        if isinstance(item, dict):
                            if "final_answer" in item:
                                pred_answer = item["final_answer"]
                            if "end_type" in item:
                                end_type = item["end_type"]
                            if "repaired_tool_call_count" in item:
                                repaired_tool_call_count = int(item["repaired_tool_call_count"])
                            if "forced_finalization_turns" in item:
                                forced_finalization_turns = int(item["forced_finalization_turns"])

                judge = judge_one_sample(pred_answer, gold_answer)

                json_line["response"] = response_res
                json_line["pred_answer"] = judge["pred_answer"]
                json_line["gold_answer"] = judge["gold_answer"]
                json_line["is_scored"] = judge["is_scored"]
                json_line["is_correct"] = judge["is_correct"]
                json_line["end_type"] = end_type
                json_line["repaired_tool_calls"] = repaired_tool_call_count
                json_line["forced_finalization_turns"] = forced_finalization_turns

                try:
                    self._set_state("write", sample_id)
                    with open(self.filename, "a+", encoding="utf-8") as f_a:
                        f_a.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                except Exception as e:
                    log_error(f"{self.worker_name} | failed to write output | {e}")
                    continue

                if judge["is_correct"] is True:
                    log_success(
                        f"{self.worker_name} | sample={sample_id} | end_type={end_type} "
                        f"| pred={judge['pred_answer']} | gold={judge['gold_answer']} | CORRECT"
                    )
                elif judge["is_correct"] is False:
                    log_warn(
                        f"{self.worker_name} | sample={sample_id} | end_type={end_type} "
                        f"| pred={judge['pred_answer']} | gold={judge['gold_answer']} | WRONG"
                    )
                else:
                    log_info(
                        f"{self.worker_name} | sample={sample_id} | end_type={end_type} "
                        f"| pred={judge['pred_answer']} | gold={judge['gold_answer']} | UNSCORED"
                    )
        finally:
            if client is not None:
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception as e:
                        log_warn(f"{self.worker_name} | client close failed | {e}")
            self._set_state("finished")
            log_info(f"{self.worker_name} | finished")


# ============================================================
# Driver
# ============================================================

def run_benchmark(
    input_dataset_dir: str,
    output_base_dir: str,
    model_name: str,
    api_base_url: str,
    api_key: str,
    thread_num: int,
    max_tool_rounds: int,
    max_total_turns: int,
    temperature: float,
    reasoning_effort: str,
    http_timeout_seconds: int,
    code_exec_timeout_seconds: int,
    worker_stall_timeout_seconds: int,
    allow_raw_tool_code_args: bool = False,
    openrouter_provider_order: Optional[str] = None,
    openrouter_provider_quantizations: Optional[str] = None,
    openrouter_allow_fallbacks: bool = True,
    run_id: Optional[str] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> None:
    resolved_run_id = sanitize_path_component(run_id) if run_id else time.strftime("%Y%m%d_%H%M%S", time.localtime())
    openrouter_provider_preferences = build_openrouter_provider_preferences(
        provider_order=openrouter_provider_order,
        provider_quantizations=openrouter_provider_quantizations,
        allow_fallbacks=openrouter_allow_fallbacks,
    )
    if client_factory is None:
        def client_factory() -> BenchmarkClient:
            return BenchmarkClient(
                model_name=model_name,
                base_url=api_base_url,
                api_key=api_key,
                http_timeout_seconds=http_timeout_seconds,
                openrouter_provider_preferences=openrouter_provider_preferences,
                verbose=True,
            )

    dataset_dir = Path(input_dataset_dir)
    output_dir = Path(output_base_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"input_dataset_dir does not exist: {dataset_dir}")

    overall_counter = init_score_counter()
    overall_file_summaries: List[Dict[str, Any]] = []

    for file_path in dataset_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".json", ".jsonl"}:
            log_warn(f"skip non-json file: {file_path}")
            continue

        log_info(f"dataset file: {file_path}")

        prefix_filename = file_path.stem
        eval_base_dir = output_dir / prefix_filename
        worker_output_dir = eval_base_dir / f"threads_res_{resolved_run_id}"
        scratch_root = eval_base_dir / f"scratch_{resolved_run_id}"

        with open(file_path, "r", encoding="utf-8") as reader_f:
            raw_lines = [line for line in reader_f if line.strip()]
        log_info(f"raw lines num: {len(raw_lines)}")

        lines: List[str] = []
        if worker_output_dir.exists():
            log_info("worker_output_dir exists, resuming unfinished prompts")
            existed_prompt_set = merge_query_set(str(worker_output_dir))
            for line in raw_lines:
                try:
                    json_line = json.loads(line)
                    prompt = json_line.get("prompt")
                    if prompt and prompt not in existed_prompt_set:
                        lines.append(line)
                except Exception as e:
                    log_warn(f"skip malformed input line: {e}")
        else:
            log_info("worker_output_dir not exists, starting fresh")
            lines = raw_lines

        log_info(f"remaining lines num: {len(lines)}")

        worker_output_dir.mkdir(parents=True, exist_ok=True)
        target_file_template = str(worker_output_dir / f"{prefix_filename}_worker*")

        task_queue = queue.Queue()
        for line in lines:
            task_queue.put(line)

        log_info(f"queue_len: {task_queue.qsize()}")
        log_info(f"worker_output_dir: {worker_output_dir}")
        log_info(f"target_file_template: {target_file_template}")

        threads = []
        worker_states: Dict[str, WorkerState] = {}
        worker_state_lock = threading.Lock()
        start_ts = time.time()

        for thread_id in range(thread_num):
            thread = BenchmarkWorker(
                thread_id=thread_id,
                task_queue=task_queue,
                target_filename_pattern=target_file_template,
                client_factory=client_factory,
                max_tool_rounds=max_tool_rounds,
                max_total_turns=max_total_turns,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                scratch_root=scratch_root,
                code_exec_timeout_seconds=code_exec_timeout_seconds,
                allow_raw_tool_code_args=allow_raw_tool_code_args,
                worker_states=worker_states,
                worker_state_lock=worker_state_lock,
            )
            time.sleep(0.1)
            thread.start()
            threads.append(thread)

        join_threads_with_monitoring(
            threads=threads,
            worker_states=worker_states,
            worker_state_lock=worker_state_lock,
            stall_timeout_seconds=worker_stall_timeout_seconds,
        )

        elapsed = time.time() - start_ts
        log_success(f"{prefix_filename} finished in {elapsed:.2f}s")

        merge_jsonl_dir = eval_base_dir / f"merge_jsonl_{resolved_run_id}"
        merge_jsonl_dir.mkdir(parents=True, exist_ok=True)
        merge_jsonl_path = merge_jsonl_dir / f"{prefix_filename}.jsonl"
        merge_jsonls(str(worker_output_dir), str(merge_jsonl_path))

        file_summary = summarize_jsonl_scores(str(merge_jsonl_path))
        show_score_table(f"Score Summary: {prefix_filename}", file_summary)

        file_counts = file_summary["counts"]
        for k in overall_counter:
            overall_counter[k] += file_counts.get(k, 0)

        overall_file_summaries.append(
            {
                "file": str(file_path),
                "merged_output": str(merge_jsonl_path),
                "elapsed_seconds": elapsed,
                "score_summary": {
                    "counts": file_summary["counts"],
                    "accuracy_all": file_summary["accuracy_all"],
                    "accuracy_scored": file_summary["accuracy_scored"],
                    "accuracy_answered": file_summary["accuracy_answered"],
                    "function_call_success_rate": file_summary["function_call_success_rate"],
                    "function_call_parse_success_rate": file_summary["function_call_parse_success_rate"],
                },
            }
        )

        show_result_panel(
            title=f"Dataset Finished: {prefix_filename}",
            body=(
                f"Input file: {file_path}\n"
                f"Run id: {resolved_run_id}\n"
                f"Thread output dir: {worker_output_dir}\n"
                f"Merged output: {merge_jsonl_path}\n"
                f"Elapsed: {elapsed:.2f}s\n"
                f"Accuracy(all): {file_summary['accuracy_all']}\n"
                f"Accuracy(scored): {file_summary['accuracy_scored']}"
            ),
            style="green",
        )

    overall_summary = {
        "counts": overall_counter,
        "accuracy_all": accuracy_str(overall_counter["correct"], overall_counter["total"]),
        "accuracy_scored": accuracy_str(overall_counter["correct"], overall_counter["scored_total"]),
        "accuracy_answered": accuracy_str(overall_counter["correct"], overall_counter["answered"]),
        "function_call_success_rate": accuracy_str(
            overall_counter["function_call_success"],
            overall_counter["function_call_attempts"],
        ),
        "function_call_parse_success_rate": accuracy_str(
            overall_counter["function_call_parse_success"],
            overall_counter["function_call_attempts"],
        ),
        "files": overall_file_summaries,
    }

    summary_path = output_dir / f"summary_{resolved_run_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    show_score_table("Overall Score Summary", overall_summary)
    show_result_panel(
        title="Overall Summary Saved",
        body=f"summary.json: {summary_path}",
        style="bright_green",
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tool-using benchmark in one merged script with scoring")

    parser.add_argument("--api_base_url", type=str, default=API_BASE_URL)
    parser.add_argument("--api_key", type=str, default=API_KEY)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)

    parser.add_argument("--input_dataset_dir", type=str, default=INPUT_DATASET_DIR)
    parser.add_argument("--output_base_dir", type=str, default=OUTPUT_BASE_DIR)

    parser.add_argument("--thread_num", type=int, default=DEFAULT_THREAD_NUM)
    parser.add_argument("--http_timeout_seconds", type=int, default=DEFAULT_HTTP_TIMEOUT_SECONDS)
    parser.add_argument("--code_exec_timeout_seconds", type=int, default=CODE_EXEC_TIMEOUT_SECONDS)
    parser.add_argument("--worker_stall_timeout_seconds", type=int, default=DEFAULT_WORKER_STALL_TIMEOUT_SECONDS)
    parser.add_argument("--allow_raw_tool_code_args", action="store_true")
    parser.add_argument("--openrouter_provider_order", type=str, default=None)
    parser.add_argument("--openrouter_provider_quantizations", type=str, default=None)
    parser.add_argument("--openrouter_disable_provider_fallbacks", action="store_true")
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--max_tool_rounds", type=int, default=MAX_TOOL_ROUNDS)
    parser.add_argument("--max_total_turns", type=int, default=MAX_TOTAL_TURNS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=DEFAULT_REASONING_EFFORT,
        choices=["low", "medium", "high"],
    )

    args = parser.parse_args()

    show_banner(
        "Tool-Use Benchmark Runner",
        "Merged driver + controller + scoring + newline/debug instrumentation",
    )
    show_config_table(
        {
            "api_base_url": args.api_base_url,
            "model_name": args.model_name,
            "input_dataset_dir": args.input_dataset_dir,
            "output_base_dir": args.output_base_dir,
            "thread_num": args.thread_num,
            "http_timeout_seconds": args.http_timeout_seconds,
            "code_exec_timeout_seconds": args.code_exec_timeout_seconds,
            "worker_stall_timeout_seconds": args.worker_stall_timeout_seconds,
            "allow_raw_tool_code_args": args.allow_raw_tool_code_args,
            "openrouter_provider_order": args.openrouter_provider_order or "<default>",
            "openrouter_provider_quantizations": args.openrouter_provider_quantizations or "<default>",
            "openrouter_allow_fallbacks": not args.openrouter_disable_provider_fallbacks,
            "run_id": args.run_id or "<auto>",
            "max_tool_rounds": args.max_tool_rounds,
            "max_total_turns": args.max_total_turns,
            "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort,
        }
    )

    if args.api_key == "REPLACE_ME":
        log_warn("API key is still REPLACE_ME. Pass --api_key or edit the constant.")

    try:
        run_benchmark(
            input_dataset_dir=args.input_dataset_dir,
            output_base_dir=args.output_base_dir,
            model_name=args.model_name,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            thread_num=args.thread_num,
            max_tool_rounds=args.max_tool_rounds,
            max_total_turns=args.max_total_turns,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            http_timeout_seconds=args.http_timeout_seconds,
            code_exec_timeout_seconds=args.code_exec_timeout_seconds,
            worker_stall_timeout_seconds=args.worker_stall_timeout_seconds,
            allow_raw_tool_code_args=args.allow_raw_tool_code_args,
            openrouter_provider_order=args.openrouter_provider_order,
            openrouter_provider_quantizations=args.openrouter_provider_quantizations,
            openrouter_allow_fallbacks=not args.openrouter_disable_provider_fallbacks,
            run_id=args.run_id,
        )
        show_result_panel("Run Complete", "All dataset files have been processed and scored.", style="bright_green")
    except Exception as e:
        log_error(f"fatal error: {type(e).__name__}: {e}")
        raise
