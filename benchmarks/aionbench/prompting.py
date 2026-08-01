"""Prompt construction and answer extraction.

The parsers here decide the difference between "wrong" and "unparseable", so
they are deliberately generous about *format* and strict about *content*: a
model that says the right thing in an unexpected shape should be scored on
what it said, but a model that says nothing scoreable must be recorded as a
structural failure rather than silently graded as wrong.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .contracts import BenchTask, SeriesRef

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

ANSWER_TAG = "ANSWER"

SYSTEM_DIRECT = (
    "You are a careful time-series analyst. Reason step by step, then commit "
    f"to exactly one final answer inside <{ANSWER_TAG}></{ANSWER_TAG}> tags. "
    "The tags must contain the answer alone — no units, no explanation, no "
    "restatement of the question. If you genuinely cannot answer, still emit "
    f"the tags containing the single word ABSTAIN."
)

SYSTEM_AION = (
    "You are a careful time-series analyst with access to Aion, a temporal "
    "reasoning runtime. Aion owns every number: it validates the data, "
    "backtests model selection, computes intervals, and abstains when the "
    "history cannot support an answer.\n\n"
    "Rules you must follow:\n"
    "1. Call Aion's tools on the supplied CSV paths before asserting anything "
    "quantitative about the series.\n"
    "2. Quote Aion's numbers verbatim. Never adjust, round beyond what Aion "
    "reported, or invent a value Aion did not produce.\n"
    "3. If Aion reports an unsupported series, say so — an abstention is a "
    "valid answer and is scored as such where the task allows it.\n"
    f"4. Finish with exactly one <{ANSWER_TAG}></{ANSWER_TAG}> block "
    "containing the answer alone."
)

SYSTEM_CODE = (
    "You are a careful time-series analyst who answers by writing Python. "
    "The named variables described in the question are already defined in "
    "your execution environment; do not redefine, regenerate, or overwrite "
    "them. Write your code inside a single <execute></execute> block and "
    "assign your result to a variable named `predictions`. Do not use "
    "markdown code fences."
)

SYSTEM_AION_CODE = (
    SYSTEM_CODE
    + "\n\nThe `aion` package is importable in your environment. Prefer "
    "`aion.forecast` over hand-rolled extrapolation: it backtests candidate "
    "models against mandatory baselines and reports a support assessment. "
    "Treat an unsupported verdict as information, not as an obstacle to route "
    "around."
)

SYSTEM_PROMPTS = {
    "direct": SYSTEM_DIRECT,
    "aion": SYSTEM_AION,
    "code": SYSTEM_CODE,
    "aion_code": SYSTEM_AION_CODE,
}


def render_series(series: SeriesRef, decimals: int = 1) -> str:
    """Series as text, one decimal place, comma separated.

    This is the textual tokenisation TimeSeriesExam evaluates (the paper's
    other arm renders the series as an image). Values are not rescaled:
    rescaling changes the shape the questions ask about.
    """
    return ", ".join(f"{value:.{decimals}f}" for value in series.values)


def render_series_block(series: Sequence[SeriesRef], decimals: int = 1, max_points: int | None = None) -> str:
    """All of a task's series, labelled, with any truncation disclosed."""
    parts: list[str] = []
    for item in series:
        values = item.values
        note = ""
        if max_points is not None and len(values) > max_points:
            values = values[-max_points:]
            note = f" (most recent {max_points} of {len(item.values)} points)"
        rendered = ", ".join(f"{value:.{decimals}f}" for value in values)
        parts.append(f"{item.name} ({len(values)} points{note}):\n[{rendered}]")
    return "\n\n".join(parts)


def render_csv_block(paths: dict[str, str], series: Sequence[SeriesRef]) -> str:
    """Describe the on-disk CSVs the Aion conditions operate over."""
    lines = [
        "The same series are on disk as CSV files with columns "
        "`timestamp,value`. Pass these paths to Aion's tools:",
    ]
    for item in series:
        path = paths.get(item.name)
        if not path:
            continue
        lines.append(
            f"- {item.name}: {path}  "
            f"(frequency {item.frequency}, {len(item.values)} rows, "
            f"time column `timestamp`, target column `value`)"
        )
    lines.append(
        "The timestamps are a regular index synthesised by the harness so the "
        "series has a temporal axis; the benchmark itself supplies only the "
        "values, in order. Treat spacing as uniform and unmeaningful."
    )
    return "\n".join(lines)


def multiple_choice_block(options: Sequence[str]) -> str:
    return "\n".join(f"{LETTERS[index]}. {option}" for index, option in enumerate(options))


def build_user_prompt(
    task: BenchTask,
    *,
    condition: str,
    csv_paths: dict[str, str] | None = None,
    include_hint: bool = False,
    include_concepts: bool = False,
    decimals: int = 1,
    max_points: int | None = None,
) -> str:
    """Assemble the user turn for a task under a condition."""
    blocks = [f"[task_id: {task.task_id}]", task.prompt.strip()]

    if task.series:
        if condition in ("aion", "aion_code") and csv_paths:
            blocks.append(render_csv_block(csv_paths, task.series))
        if condition != "code" or not csv_paths:
            blocks.append("Series values:\n" + render_series_block(task.series, decimals, max_points))

    if task.options:
        blocks.append("Options:\n" + multiple_choice_block(task.options))

    if include_concepts and task.concepts:
        blocks.append("Relevant concepts:\n" + "\n".join(f"- {item}" for item in task.concepts))
    if include_hint and task.hint:
        blocks.append(f"Hint: {task.hint}")

    if task.constraint:
        blocks.append(f"Constraint that the answer must satisfy: {task.constraint}")

    if task.kind == "multiple_choice":
        blocks.append(
            f"Reply with the letter of the single best option inside "
            f"<{ANSWER_TAG}></{ANSWER_TAG}> tags."
        )
    elif task.kind == "numeric":
        blocks.append(f"Reply with the numeric answer alone inside <{ANSWER_TAG}></{ANSWER_TAG}> tags.")
    elif task.kind == "series":
        blocks.append(
            f"Reply with a JSON array of the requested values inside "
            f"<{ANSWER_TAG}></{ANSWER_TAG}> tags — no prose inside the tags."
        )

    return "\n\n".join(block for block in blocks if block)


# -- answer extraction -----------------------------------------------------

_TAG_RE = re.compile(rf"<{ANSWER_TAG}>(.*?)</{ANSWER_TAG}>", re.IGNORECASE | re.DOTALL)
_LOOSE_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is)?\s*[:\-]\s*\(?([A-Z])\)?\b", re.IGNORECASE
)
_BARE_LETTER_RE = re.compile(r"^\s*\(?([A-Z])\)?[.):]?\s*$")
_EXECUTE_RE = re.compile(r"<execute>(.*?)</execute>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

_ABSTAIN_WORDS = ("abstain", "unsupported", "cannot answer", "insufficient data")


def extract_tagged(text: str) -> str | None:
    """The content of the last well-formed answer block, if any."""
    matches = _TAG_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def is_abstention(text: str | None) -> bool:
    lowered = (text or "").strip().lower()
    return any(word in lowered for word in _ABSTAIN_WORDS)


def parse_choice(text: str, option_count: int) -> int | None:
    """Index of the chosen option, or None when nothing parseable was said.

    Tried in order of decreasing confidence: the answer block, an explicit
    "Answer: X", a final line that is just a letter. Anything that resolves
    outside the option range is rejected rather than clamped.
    """
    if option_count <= 0:
        return None
    valid = LETTERS[:option_count]

    tagged = extract_tagged(text)
    for candidate in (tagged, text):
        if not candidate:
            continue
        index = _letter_index(candidate.strip(), valid)
        if index is not None:
            return index

    match = _LOOSE_ANSWER_RE.search(text or "")
    if match and match.group(1).upper() in valid:
        return valid.index(match.group(1).upper())

    for line in reversed((text or "").strip().splitlines()):
        bare = _BARE_LETTER_RE.match(line)
        if bare and bare.group(1).upper() in valid:
            return valid.index(bare.group(1).upper())
    return None


def _letter_index(candidate: str, valid: str) -> int | None:
    if not candidate:
        return None
    stripped = candidate.strip().strip("().: \t\"'*")
    if len(stripped) == 1 and stripped.upper() in valid:
        return valid.index(stripped.upper())
    # "Option B", "B. rising trend", "(C)"
    match = re.match(r"^(?:option\s+)?\(?([A-Z])\)?\b", stripped, re.IGNORECASE)
    if match and match.group(1).upper() in valid:
        return valid.index(match.group(1).upper())
    return None


def parse_number(text: str) -> float | None:
    """The single number the model committed to, or None."""
    tagged = extract_tagged(text)
    for candidate in (tagged, text):
        if not candidate:
            continue
        numbers = _NUMBER_RE.findall(candidate.replace(",", ""))
        if numbers:
            try:
                # Inside the answer block the first number is the answer; in
                # free prose the last one is what the model settled on.
                return float(numbers[0] if candidate is tagged else numbers[-1])
            except ValueError:
                continue
    return None


def parse_series(text: str) -> list[float] | None:
    """A JSON array (possibly nested) flattened to floats, or None."""
    tagged = extract_tagged(text) or text or ""
    for blob in _json_candidates(tagged):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        flat = _flatten_numbers(parsed)
        if flat:
            return flat
    return None


def parse_matrix(text: str) -> list[list[float]] | None:
    """A 2-D JSON array, preserving shape. Rows of unequal length are rejected."""
    tagged = extract_tagged(text) or text or ""
    for blob in _json_candidates(tagged):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list) or not parsed:
            continue
        if all(isinstance(row, list) for row in parsed):
            rows = [_flatten_numbers(row) for row in parsed]
            if rows and all(rows) and len({len(row) for row in rows}) == 1:
                return rows
            # Nested but ragged: not a matrix. Flattening it here would
            # silently invent a shape, so leave it to the flat parser.
            continue
        flat = _flatten_numbers(parsed)
        if flat:
            return [flat]
    return None


def parse_code(text: str) -> str | None:
    """The Python the model wants executed, from <execute> or a fenced block."""
    match = _EXECUTE_RE.findall(text or "")
    if match:
        return match[-1].strip()
    fenced = _FENCE_RE.findall(text or "")
    return fenced[-1].strip() if fenced else None


def _json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start:end + 1])
    return [item for item in candidates if item]


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numbers(item))
        return out
    return []
