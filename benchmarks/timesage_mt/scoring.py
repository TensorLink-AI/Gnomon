"""Score agent responses with TimeSage-MT's dataset-embedded verify specs.

Each reference agent turn in the official task files carries a
``finding_verify`` object. The mechanical rule, stated precisely:
listed ``keywords`` are always checked, whatever the spec's ``type``
says; a numerical ``range`` is checked only when ``type`` is
``numerical_range``, ``keyword``, or absent. Any spec where at least
one of those checks applies is graded mechanically on those parts alone
— e.g. an ``embedding_threshold`` spec that also lists keywords is
graded on the keywords only, its threshold ignored. Only specs where
neither check applies fall through to the judge path: they are counted
as ``unscored`` unless a judge model is supplied, and judge-scored
turns are reported separately because the official platform's judge is
not public.

The checks themselves:

- keywords — every listed keyword must appear in the response
  (case-insensitive substring).
- range — some number in the response must fall inside the inclusive
  ``[low, high]`` range (after keyword checks, if both given). Numbers
  that are components of date/timestamp tokens (``2026-01-02``,
  ``2026/01/02``, ``12:30:05``, ISO datetimes) are excluded from
  extraction, identically for both arms.

Remaining range leniency, disclosed: ANY in-range number anywhere in
the response passes, and the tools-arm prompt elicits more numbers per
response, so the check's expected benefit scales with response
numerosity. Whether the official judge shares this leniency is not
knowable from the published dataset.
"""

from __future__ import annotations

import re
from typing import Any

NUMBER_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?")

# Date/timestamp tokens whose digit components must not count as answer
# numbers: calendar dates (2026-01-02, 2026/01/02), clock times
# (12:30, 12:30:05.5), and full ISO datetimes with optional Z/offset.
DATETIME_PATTERN = re.compile(
    r"(?<![\d.])(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r"|\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
    r")(?!\d)"
)

JUDGE_PROMPT = """\
You are grading one turn of a time-series analysis dialogue.

Reference key finding:
{key_finding}

{rubric_block}
Agent response to grade:
<response>
{response}
</response>

Does the response communicate the key finding correctly (numbers of the
same magnitude, correct direction/period/conclusion)? Answer with only
PASS or FAIL.
"""


def numbers_in(text: str) -> list[float]:
    values = []
    for match in NUMBER_PATTERN.findall(DATETIME_PATTERN.sub(" ", text)):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def score_mechanical(verify: dict[str, Any], response: str) -> bool | None:
    """Apply the mechanical parts of one finding_verify spec.

    Returns True/False when the spec is mechanically checkable, or None
    when it requires embedding/judge evaluation.
    """
    verify_type = verify.get("type")
    keywords = verify.get("keywords") or []
    value_range = verify.get("range")

    checks: list[bool] = []
    if keywords:
        lowered = response.lower()
        checks.append(all(str(k).lower() in lowered for k in keywords))
    if value_range and len(value_range) == 2 and verify_type in (
        "numerical_range", "keyword", None
    ):
        low, high = float(value_range[0]), float(value_range[1])
        checks.append(any(low <= v <= high for v in numbers_in(response)))
    if checks:
        return all(checks)
    return None


def score_with_judge(
    verify: dict[str, Any], reference_turn: dict[str, Any],
    response: str, judge_client: Any,
) -> bool:
    rubric = verify.get("judge_rubric")
    prompt = JUDGE_PROMPT.format(
        key_finding=reference_turn.get("key_finding") or "(see rubric)",
        rubric_block=f"Grading rubric:\n{rubric}\n" if rubric else "",
        response=response[:8000],
    )
    completion = judge_client.completions(
        [{"role": "user", "content": prompt}], n=1
    )[0]
    return "PASS" in completion.upper().split("FAIL")[0] if "FAIL" in completion.upper() \
        else "PASS" in completion.upper()


def score_turn(
    reference_turn: dict[str, Any], response: str,
    judge_client: Any | None = None,
) -> dict[str, Any]:
    """Score one agent response against its reference turn.

    Returns ``{"scored": bool, "passed": bool | None, "basis": str}``.
    """
    verify = reference_turn.get("finding_verify")
    if not verify:
        return {"scored": False, "passed": None, "basis": "no_verify_spec"}
    mechanical = score_mechanical(verify, response)
    if mechanical is not None:
        return {"scored": True, "passed": mechanical, "basis": "mechanical"}
    if judge_client is not None:
        passed = score_with_judge(verify, reference_turn, response, judge_client)
        return {"scored": True, "passed": passed, "basis": "llm_judge"}
    return {"scored": False, "passed": None, "basis": "needs_judge"}
