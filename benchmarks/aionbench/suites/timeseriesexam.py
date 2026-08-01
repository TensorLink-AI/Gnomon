"""TimeSeriesExam — do models understand basic time-series concepts?

Cai, Choudhry, Goswami and Dubrawski, *TimeSeriesExam: A Time Series
Understanding Exam* (arXiv:2410.14752, NeurIPS Workshop on Time Series in the
Age of Large Models). 746 procedurally generated multiple-choice questions
over five categories — pattern recognition, noise understanding, similarity /
comparative reasoning, anomaly detection, and causality — refined with Item
Response Theory so each item discriminates between models of different
ability.

The paper's headline finding is that models handle simple pattern questions
and fall apart on causality. That makes it the sharpest of the three suites
for the question this repo cares about: causality and anomaly items are
exactly where a deterministic detector should beat a model's intuition, and
the per-category breakdown says whether Aion's tools moved the categories
they should have moved rather than just the aggregate.

Faithfulness notes:

- The paper evaluates a text arm and an image arm; the text arm renders
  values to one decimal place, comma separated, unscaled. This harness runs
  the text arm — that is the arm a tool-augmented condition is comparable to.
- Hints and relevant-concept blocks are off by default. The paper reports
  that concepts can *hurt*; both are exposed as flags so that ablation is
  reproducible here.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts import Attempt, BenchTask, SeriesRef, TaskOutcome
from ..datasets import fetch, hf_url, read_table
from ..prompting import LETTERS, is_abstention, parse_choice
from .base import LoadOptions, ScoringContext, Suite, register

REPO_ID = "AutonLab/TimeSeriesExam1"
DATA_FILE = "data/test-00000-of-00001.parquet"

# The dataset's own labels, mapped onto the five categories the paper reports.
_CATEGORY_ALIASES = {
    "pattern": "pattern_recognition",
    "pattern recognition": "pattern_recognition",
    "noise": "noise_understanding",
    "noise understanding": "noise_understanding",
    "similarity": "similarity_analysis",
    "similarity analysis": "similarity_analysis",
    "comparative": "similarity_analysis",
    "comparative reasoning": "similarity_analysis",
    "anomaly": "anomaly_detection",
    "anomaly detection": "anomaly_detection",
    "causality": "causality_analysis",
    "causality analysis": "causality_analysis",
    "causal": "causality_analysis",
}

CATEGORIES = (
    "pattern_recognition",
    "noise_understanding",
    "similarity_analysis",
    "anomaly_detection",
    "causality_analysis",
)


@register
class TimeSeriesExamSuite(Suite):
    name = "timeseriesexam"
    description = (
        "746 IRT-refined multiple-choice questions on time-series concepts "
        "across five categories (arXiv:2410.14752)."
    )
    citation = "Cai et al., TimeSeriesExam: A Time Series Understanding Exam, arXiv:2410.14752"
    homepage = "https://huggingface.co/datasets/AutonLab/TimeSeriesExam1"
    conditions = ("direct", "aion")

    def load(self, options: LoadOptions) -> list[BenchTask]:
        path = options.data_path or fetch(
            hf_url(REPO_ID, DATA_FILE, revision=options.revision),
            cache=options.cache, filename="timeseriesexam-test.parquet",
        )
        rows = read_table(path)
        tasks = [task for task in (self._to_task(index, row) for index, row in enumerate(rows)) if task]
        return self.subsample(tasks, options)

    def _to_task(self, index: int, row: Mapping[str, Any]) -> BenchTask | None:
        options_list = _string_list(row.get("options"))
        if len(options_list) < 2:
            return None

        answer_index = _answer_index(row.get("answer"), options_list)
        if answer_index is None:
            return None

        series = _series_from_row(row)
        if not series:
            return None

        identifier = row.get("id")
        task_id = f"tse-{identifier}" if identifier is not None else f"tse-{index:05d}"

        return BenchTask(
            task_id=task_id,
            suite=self.name,
            kind="multiple_choice",
            prompt=str(row.get("question") or "").strip(),
            series=series,
            options=tuple(options_list),
            answer=LETTERS[answer_index],
            metric="accuracy",
            category=_normalise_category(row.get("category")),
            subcategory=str(row.get("subcategory") or "").strip(),
            difficulty=str(row.get("difficulty") or "").strip(),
            hint=str(row.get("question_hint") or "").strip(),
            concepts=tuple(_string_list(row.get("relevant_concepts"))),
            metadata={
                "template_id": row.get("tid"),
                "question_type": row.get("question_type"),
                "format_hint": row.get("format_hint"),
                "series_count": len(series),
            },
        )

    def score(self, task: BenchTask, attempt: Attempt, context: ScoringContext) -> TaskOutcome:
        transport = self.transport_failure(task, attempt)
        if transport is not None:
            return transport

        outcome = self.new_outcome(task, attempt)
        outcome.expected = task.answer

        chosen = parse_choice(attempt.text, len(task.options))
        if chosen is None:
            outcome.correct = False
            outcome.failure = "refusal" if is_abstention(attempt.text) else "structural"
            outcome.detail = (
                "No option letter could be recovered from the response."
                if outcome.failure == "structural" else "Model declined to choose."
            )
            return outcome

        answered = LETTERS[chosen]
        outcome.answered = answered
        outcome.correct = answered == task.answer
        outcome.score = 1.0 if outcome.correct else 0.0
        if not outcome.correct:
            outcome.failure = "threshold"
            outcome.detail = f"Chose {answered}, expected {task.answer}."
        return outcome


# -- row parsing -----------------------------------------------------------

def _series_from_row(row: Mapping[str, Any]) -> tuple[SeriesRef, ...]:
    """Pull whichever of ts / ts1 / ts2 the row actually populated.

    Single-series items use ``ts``; comparative and causality items use
    ``ts1`` and ``ts2``. Rows carry all three columns with the unused ones
    empty, so presence has to be tested rather than assumed from the category.
    """
    series: list[SeriesRef] = []
    for column, label in (("ts", "series"), ("ts1", "series_1"), ("ts2", "series_2")):
        values = _float_list(row.get(column))
        if len(values) >= 2:
            series.append(SeriesRef(name=label, values=tuple(values)))
    return tuple(series)


def _answer_index(answer: Any, options: Sequence[str]) -> int | None:
    """Resolve the dataset's answer field to an option index.

    The field is a bare string that has appeared as a letter, an integer
    index, and the full option text across dataset revisions. All three are
    accepted; anything else is dropped rather than guessed at, because a
    mis-keyed answer silently corrupts every model's score.
    """
    if answer is None:
        return None
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer if 0 <= answer < len(options) else None

    text = str(answer).strip()
    if not text:
        return None

    stripped = text.strip("().: \t'\"")
    if len(stripped) == 1 and stripped.upper() in LETTERS[:len(options)]:
        return LETTERS.index(stripped.upper())
    if stripped.isdigit():
        index = int(stripped)
        if 0 <= index < len(options):
            return index
        if 1 <= index <= len(options):     # 1-based variants
            return index - 1
        return None

    normalised = _normalise(text)
    for index, option in enumerate(options):
        if _normalise(option) == normalised:
            return index
    return None


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split()).strip(" .")


def _normalise_category(raw: Any) -> str:
    text = _normalise(str(raw or ""))
    if not text:
        return "uncategorised"
    if text in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[text]
    collapsed = text.replace("-", " ").replace("_", " ")
    if collapsed in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[collapsed]
    for prefix, mapped in _CATEGORY_ALIASES.items():
        if collapsed.startswith(prefix):
            return mapped
    return collapsed.replace(" ", "_")


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    try:
        items = list(value)
    except TypeError:
        return []
    out: list[float] = []
    for item in items:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            import ast

            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return [text]
            return [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else [text]
        return [text] if text else []
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]
