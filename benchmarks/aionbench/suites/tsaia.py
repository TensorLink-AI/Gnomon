"""TSAIA — can an LLM act as a time-series AI assistant?

Ye, Liu, Cao, Yang and Liu, *When LLM Meets Time Series: Can LLMs Perform
Multi-Step Time Series Reasoning and Inference* (arXiv:2509.01822). 1,054
questions over 33 real-world task formulations drawn from 20+ publications,
grouped into predictive, diagnostic, analytical, and financial
decision-making families, over PSML, ERA5, MIT-BIH, and Yahoo Finance data.

Two subsets, and they behave very differently here:

``multiple_choice`` (150) is self-contained — the series arrive as text in
``data_info`` and the key is a letter in the row. It runs offline after one
CSV download and needs neither pickles nor code execution, which makes it the
suite to reach for first.

``analysis_questions`` (904) is the real test and the expensive one. Its
executor variables, contexts, constraints, and ground truth ship as pickle
files, and its native protocol has the model answer by writing Python against
pre-loaded variables. Both of those are opt-in here: ``--allow-pickle`` to
read the assets, ``--allow-code-execution`` to run the answers.

Where this harness departs from the paper, and why:

- The paper's protocol is CodeAct throughout. This harness also offers the
  non-code conditions (``direct``, ``aion``) so a tool-augmented run can be
  compared against a code-writing one on identical items. Numbers from the
  non-code conditions are *not* comparable to the paper's table; the run
  manifest records which condition produced them.
- The paper uses task-specific success criteria with per-task bars. Those
  bars are not published per task type, so this harness applies a documented
  default per metric family, overridable in the run config. Every report
  states the bar it used. Treat cross-paper comparisons of the pass rate as
  unsound; the within-run control-vs-treatment comparison is the sound one.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Mapping, Sequence

from ..contracts import Attempt, BenchTask, SeriesRef, TaskOutcome
from ..datasets import fetch, hf_url, load_pickle, read_table
from ..metrics import binary_f1, mape, smape
from ..prompting import (
    LETTERS,
    is_abstention,
    parse_choice,
    parse_code,
    parse_matrix,
    parse_number,
    parse_series,
)
from ..sandbox import SandboxDisabled, run_code
from .base import LoadOptions, ScoringContext, Suite, register

REPO_ID = "Melady/TSAIA"
CONFIG_FILES = {
    "multiple_choice": "multiple_choice.csv",
    "analysis_questions": "analysis_questions.csv",
}

# Metric family per question-type keyword. The taxonomy is 33 task types and
# explicitly extensible, so this matches on intent rather than enumerating
# names that will drift; `--metric-overrides` pins any type that lands wrong.
_METRIC_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("anomaly", "outlier", "arrhythmia", "detect"), "f1"),
    (("classif", "label", "diagnos"), "accuracy"),
    (("imputation", "missing", "interpol"), "mape"),
    (("var", "risk", "sharpe", "drawdown", "return", "volatil"), "relative_error"),
    (("price", "forecast", "predict", "future", "load", "demand", "temperature"), "mape"),
)

# Default pass bars. Deliberately conservative and, more importantly,
# *reported*: a benchmark whose thresholds are invisible is a benchmark whose
# numbers cannot be argued with.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "mape": 0.15,             # <= 15% mean absolute percentage error
    "smape": 0.15,
    "relative_error": 0.10,   # scalar answers within 10%
    "f1": 0.50,
    "accuracy": 1.0,
}


@register
class TSAIASuite(Suite):
    name = "tsaia"
    description = (
        "1,054 constraint-aware time-series assistant tasks across 33 task "
        "types — forecasting, anomaly detection, and financial decisions "
        "(arXiv:2509.01822)."
    )
    citation = "Ye et al., When LLM Meets Time Series, arXiv:2509.01822"
    homepage = "https://huggingface.co/datasets/Melady/TSAIA"
    conditions = ("direct", "aion", "code", "aion_code")

    def load(self, options: LoadOptions) -> list[BenchTask]:
        config = options.config or "multiple_choice"
        if config not in CONFIG_FILES:
            from ..contracts import BenchError

            raise BenchError(
                "UNKNOWN_TSAIA_CONFIG",
                f"Unknown TSAIA config {config!r}.",
                {"supported": sorted(CONFIG_FILES)},
            )

        path = options.data_path or fetch(
            hf_url(REPO_ID, CONFIG_FILES[config], revision=options.revision),
            cache=options.cache, filename=f"tsaia-{config}.csv",
        )
        rows = read_table(path)
        builder = self._multiple_choice_task if config == "multiple_choice" else self._analysis_task
        tasks = [task for task in (builder(index, row, options) for index, row in enumerate(rows)) if task]
        return self.subsample(tasks, options)

    # -- loading -----------------------------------------------------------

    def _multiple_choice_task(self, index: int, row: Mapping[str, Any], options: LoadOptions) -> BenchTask | None:
        choices = _literal_list(row.get("options"))
        answer = str(row.get("answer") or "").strip().upper()
        if len(choices) < 2 or answer not in LETTERS[:len(choices)]:
            return None

        question_type = str(row.get("question_type") or "").strip()
        return BenchTask(
            task_id=f"tsaia-mc-{row.get('question_id', index)}",
            suite=self.name,
            kind="multiple_choice",
            prompt=_strip_protocol_boilerplate(str(row.get("prompt") or "")),
            series=_series_from_text(str(row.get("data_info") or "")),
            options=tuple(choices),
            answer=answer,
            metric="accuracy",
            category=_family(question_type),
            subcategory=question_type,
            metadata={
                "config": "multiple_choice",
                "question_type": question_type,
                "answer_info": row.get("answer_info"),
                "executor_variables": row.get("executor_variables"),
            },
        )

    def _analysis_task(self, index: int, row: Mapping[str, Any], options: LoadOptions) -> BenchTask | None:
        question_type = str(row.get("question_type") or "").strip()
        metric = _metric_for(question_type)

        truth_ref = row.get("ground_truth_data")
        target = None
        if truth_ref and options.allow_pickle:
            target = _load_asset(str(truth_ref), options)

        constraint = None
        constraint_ref = row.get("constraint")
        if constraint_ref and options.allow_pickle:
            constraint = _load_asset(str(constraint_ref), options)

        return BenchTask(
            task_id=f"tsaia-aq-{row.get('question_id', index)}",
            suite=self.name,
            kind="series" if metric in ("mape", "smape", "f1") else "numeric",
            prompt=_strip_protocol_boilerplate(str(row.get("prompt") or "")),
            series=_series_from_text(str(row.get("data_str") or "")),
            target=target,
            metric=metric,
            category=_family(question_type),
            subcategory=question_type,
            constraint=_describe_constraint(constraint),
            metadata={
                "config": "analysis_questions",
                "question_type": question_type,
                "executor_variables": row.get("executor_variables"),
                "ground_truth_data": truth_ref,
                "context": row.get("context"),
                "constraint_ref": constraint_ref,
                "ground_truth_loaded": target is not None,
            },
        )

    # -- scoring -----------------------------------------------------------

    def score(self, task: BenchTask, attempt: Attempt, context: ScoringContext) -> TaskOutcome:
        transport = self.transport_failure(task, attempt)
        if transport is not None:
            return transport

        if task.kind == "multiple_choice":
            return self._score_choice(task, attempt)
        return self._score_analysis(task, attempt, context)

    def _score_choice(self, task: BenchTask, attempt: Attempt) -> TaskOutcome:
        outcome = self.new_outcome(task, attempt)
        outcome.expected = task.answer
        chosen = parse_choice(attempt.text, len(task.options))
        if chosen is None:
            outcome.correct = False
            outcome.failure = "refusal" if is_abstention(attempt.text) else "structural"
            outcome.detail = "No option letter in the response."
            return outcome
        outcome.answered = LETTERS[chosen]
        outcome.correct = outcome.answered == task.answer
        outcome.score = 1.0 if outcome.correct else 0.0
        if not outcome.correct:
            outcome.failure = "threshold"
            outcome.detail = f"Chose {outcome.answered}, expected {task.answer}."
        return outcome

    def _score_analysis(self, task: BenchTask, attempt: Attempt, context: ScoringContext) -> TaskOutcome:
        outcome = self.new_outcome(task, attempt)

        if task.target is None:
            outcome.correct = None
            outcome.failure = "structural"
            outcome.detail = (
                "Ground truth was not loaded. analysis_questions ships its key "
                "as a pickle; rerun with --allow-pickle to grade this subset."
            )
            return outcome

        expected = _flatten(task.target)
        outcome.expected = expected[:32]

        answered, failure, detail = self._extract_answer(task, attempt, context)
        if failure:
            outcome.correct = False
            outcome.failure = failure
            outcome.detail = detail
            return outcome

        outcome.answered = answered[:32]
        metric = _reconcile_metric(task.metric, expected)
        outcome.metric = metric
        threshold = context.thresholds.get(metric, DEFAULT_THRESHOLDS.get(metric, 0.15))
        score, correct, note = _grade(metric, expected, answered, threshold)
        outcome.score = score
        outcome.correct = correct
        if correct is False:
            outcome.failure = "threshold"
        outcome.detail = note
        return outcome

    def _extract_answer(
        self, task: BenchTask, attempt: Attempt, context: ScoringContext,
    ) -> tuple[list[float], str | None, str]:
        """Recover a numeric answer, executing code when the condition says so."""
        if attempt.condition in ("code", "aion_code"):
            code = parse_code(attempt.text)
            if not code:
                return [], "structural", "No <execute> block in the response."
            variables_ref = task.metadata.get("executor_variables")
            variables_path = None
            if variables_ref and context.allow_pickle:
                # Handed to the sandbox as a path: the harness process never
                # unpickles the dataset's executor variables itself.
                variables_path = str(fetch(hf_url(REPO_ID, str(variables_ref))))
            try:
                result = run_code(code, variables_path=variables_path, config=context.sandbox)
            except SandboxDisabled as exc:
                return [], "execution", str(exc)
            if not result.ok:
                return [], "execution", (result.error or "Execution failed.")[:600]
            values = _flatten(result.value)
            if not values:
                return [], "structural", "`predictions` held nothing numeric."
            return values, None, ""

        if is_abstention(attempt.text):
            return [], "refusal", "Model abstained."

        matrix = parse_matrix(attempt.text)
        if matrix:
            return [value for row in matrix for value in row], None, ""
        series = parse_series(attempt.text)
        if series:
            return series, None, ""
        number = parse_number(attempt.text)
        if number is not None:
            return [number], None, ""
        return [], "structural", "No numeric answer could be recovered."


# -- grading ---------------------------------------------------------------

def _reconcile_metric(metric: str, expected: list[float]) -> str:
    """Let the ground truth's shape overrule the task-name heuristic.

    ``_metric_for`` guesses from 33 extensible task-type strings and will
    occasionally guess wrong. The key's shape cannot: a one-value answer is a
    scalar question and a multi-value answer is not, whatever the name said.
    Grading a vector question as a scalar would fail it on shape alone.
    """
    if len(expected) == 1 and metric in ("mape", "smape"):
        return "relative_error"
    if len(expected) > 1 and metric == "relative_error":
        return "mape"
    return metric


def _grade(
    metric: str, expected: list[float], answered: list[float], threshold: float,
) -> tuple[float | None, bool | None, str]:
    if not expected:
        return None, None, "Empty ground truth."

    if metric == "f1":
        length = len(expected)
        predicted = answered[:length] + [0.0] * max(0, length - len(answered))
        score = binary_f1(
            [1 if value else 0 for value in expected],
            [1 if value else 0 for value in predicted],
        ).f1
        return round(score, 6), score >= threshold, f"F1 {score:.3f} against a bar of {threshold:.2f}."

    if metric == "relative_error":
        if len(answered) != 1 or len(expected) != 1:
            # A scalar question answered with a vector is a shape failure, not
            # a near miss; grading its first element would launder that.
            return None, False, (
                f"Expected a single value, got {len(answered)} "
                f"against {len(expected)} ground-truth values."
            )
        denominator = abs(expected[0])
        if denominator < 1e-9:
            error = abs(answered[0] - expected[0])
        else:
            error = abs(answered[0] - expected[0]) / denominator
        return round(error, 6), error <= threshold, (
            f"Relative error {error:.4f} against a bar of {threshold:.2f}."
        )

    if len(answered) != len(expected):
        return None, False, (
            f"Shape mismatch: expected {len(expected)} values, got {len(answered)}."
        )
    if any(math.isnan(value) or math.isinf(value) for value in answered):
        return None, False, "Answer contained NaN or infinity."

    error = mape(expected, answered)
    label = "MAPE"
    if math.isnan(error):
        error = smape(expected, answered)
        label = "sMAPE"
    return round(error, 6), error <= threshold, (
        f"{label} {error:.4f} against a bar of {threshold:.2f}."
    )


# -- row helpers -----------------------------------------------------------

_ARRAY_RE = re.compile(r"\[([0-9eE.,\s\-+]+)\]")
_NAME_RE = re.compile(r"(?:data|series|values)\s+(?:of|for)\s+([A-Za-z0-9._-]{1,32})", re.IGNORECASE)


def _series_from_text(text: str) -> tuple[SeriesRef, ...]:
    """Pull the bracketed numeric arrays out of TSAIA's prose data block.

    TSAIA embeds its series in the prompt text — "The historical stock value
    data of TBLA for the past 19 hours is: [3.28, 3.28, ...]". Extracting them
    is what lets the Aion conditions put the same numbers on disk; when
    nothing parses, the task still runs, just without CSVs.

    The name is taken from the segment of prose immediately preceding each
    array, never from earlier ones: inheriting a neighbour's ticker would
    mislabel data, which is worse than an anonymous ``series_2``.
    """
    series: list[SeriesRef] = []
    cursor = 0
    for index, match in enumerate(_ARRAY_RE.finditer(text or "")):
        values = _parse_bracketed(match.group(1))
        segment = (text or "")[cursor:match.start()]
        cursor = match.end()
        if len(values) < 4:
            continue
        found = _NAME_RE.findall(segment)
        name = found[-1].strip(" .") if found else f"series_{index + 1}"
        series.append(SeriesRef(name=name or f"series_{index + 1}", values=tuple(values), frequency="h"))
    # Distinct names keep the materialised CSV filenames from colliding.
    seen: dict[str, int] = {}
    deduped: list[SeriesRef] = []
    for item in series:
        count = seen.get(item.name, 0)
        seen[item.name] = count + 1
        label = item.name if count == 0 else f"{item.name}_{count + 1}"
        deduped.append(SeriesRef(name=label, values=item.values, frequency=item.frequency))
    return tuple(deduped)


def _parse_bracketed(blob: str) -> list[float]:
    values: list[float] = []
    for piece in blob.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            values.append(float(piece))
        except ValueError:
            return []
    return values


_PROTOCOL_NOISE = (
    "You should enclose your python code",
    "Do not use any other tags",
    "Store your output in the variable",
    "Do not customly define/generate/overwrite",
)


def _strip_protocol_boilerplate(prompt: str) -> str:
    """Remove TSAIA's CodeAct instructions from the question text.

    The harness states the answer protocol itself, per condition. Leaving the
    dataset's own "write <execute> blocks" instructions in a non-code
    condition would tell the model to do something the condition cannot do,
    and would score the resulting confusion as a reasoning failure.
    """
    lines = [
        line for line in (prompt or "").splitlines()
        if not any(marker in line for marker in _PROTOCOL_NOISE)
    ]
    joined = "\n".join(lines)
    for marker in _PROTOCOL_NOISE:
        index = joined.find(marker)
        if index >= 0:
            joined = joined[:index]
    return joined.strip()


def _literal_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else []


def _metric_for(question_type: str) -> str:
    lowered = question_type.lower()
    for keywords, metric in _METRIC_RULES:
        if any(keyword in lowered for keyword in keywords):
            return metric
    return "mape"


def _family(question_type: str) -> str:
    """Coarse grouping matching the paper's four task categories.

    Order matters: forecasting stock prices is a *predictive* task that
    happens to use financial data, not a financial decision. The decision
    bucket is the one where the answer is an allocation or a risk figure.
    """
    lowered = question_type.lower()
    if any(word in lowered for word in ("anomaly", "outlier", "arrhythmia", "detect")):
        return "diagnostic"
    if any(word in lowered for word in ("price", "forecast", "predict", "future", "load", "demand")):
        return "predictive"
    if any(word in lowered for word in ("portfolio", "var", "sharpe", "drawdown", "trading", "allocat", "return")):
        return "financial_decision"
    return "analytical"


def _describe_constraint(constraint: Any) -> str | None:
    if constraint is None:
        return None
    if isinstance(constraint, str):
        return constraint.strip() or None
    if isinstance(constraint, Mapping):
        return "; ".join(f"{key}: {value}" for key, value in constraint.items()) or None
    if isinstance(constraint, Sequence):
        return "; ".join(str(item) for item in constraint) or None
    return str(constraint)


def _load_asset(relative: str, options: LoadOptions) -> Any:
    path = fetch(hf_url(REPO_ID, relative, revision=options.revision), cache=options.cache)
    return load_pickle(path, allowed=options.allow_pickle)


def _flatten(value: Any) -> list[float]:
    """Coerce a ground truth or answer of any nesting into a flat float list."""
    if value is None:
        return []
    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        try:
            return [float(value)]
        except ValueError:
            return []
    tolist = getattr(value, "tolist", None)     # numpy arrays, pandas objects
    if callable(tolist):
        try:
            return _flatten(tolist())
        except Exception:  # noqa: BLE001 - unknown third-party object
            return []
    values_attr = getattr(value, "values", None)   # pandas DataFrame / Series
    if values_attr is not None and not callable(values_attr):
        return _flatten(values_attr)
    if isinstance(value, Mapping):
        out: list[float] = []
        for item in value.values():
            out.extend(_flatten(item))
        return out
    if isinstance(value, Sequence):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return []
