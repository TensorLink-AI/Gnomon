"""Agentic time-series question answering, after TimeART.

Wu et al., *TimeART: Towards Agentic Time Series Reasoning via
Tool-Augmentation* (arXiv:2601.13653). TimeART's claim is that a language
model paired with strong out-of-the-box time-series tools behaves like a data
scientist on Time Series Question Answering, beating open-source baselines
and holding its own against much larger closed models. It trains its own
reasoning models on a 100k expert-trajectory corpus (TimeToolBench) and
reports n-way accuracy on MTBench and TimeMQA.

This suite runs the *ablation* at the centre of that claim — the same model,
the same questions, with and without a real tool surface — using Aion as the
toolbelt. It does not reproduce TimeART's training pipeline, and it does not
ship TimeART's weights; what it measures is whether tool augmentation moves
the accuracy on a TSQA corpus you point it at.

Because the TSQA corpora are separate datasets on their own release
schedules, this suite is corpus-agnostic: it reads a normalised JSONL/parquet
row format and takes a field map for anything shaped differently. A small
offline fixture ships with the package so the suite — and every code path
around it — runs end to end with no network and no key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import Attempt, BenchError, BenchTask, SeriesRef, TaskOutcome
from ..datasets import fetch, hf_url, read_table
from ..prompting import LETTERS, is_abstention, parse_choice
from .base import LoadOptions, ScoringContext, Suite, register

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "tsqa_smoke.jsonl"

DEFAULT_FIELD_MAP = {
    "id": "id",
    "question": "question",
    "options": "options",
    "answer": "answer",
    "series": "series",
    "category": "category",
    "context": "context",
}

# Field maps for corpora whose column names differ. Add a corpus here rather
# than teaching the loader about it: everything else in the suite is generic.
CORPUS_PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "path": str(FIXTURE),
        "field_map": dict(DEFAULT_FIELD_MAP),
    },
    "generic": {
        "field_map": dict(DEFAULT_FIELD_MAP),
    },
    "mtbench": {
        # Set repo/file via --dataset-repo / --dataset-file (or extra.path);
        # the column names below are the ones the loader will look for.
        "field_map": {
            "id": "id",
            "question": "question",
            "options": "options",
            "answer": "answer",
            "series": "time_series",
            "category": "task",
            "context": "news",
        },
    },
    "timemqa": {
        "field_map": {
            "id": "qid",
            "question": "question",
            "options": "choices",
            "answer": "answer",
            "series": "series",
            "category": "type",
            "context": "context",
        },
    },
}


@register
class TSQASuite(Suite):
    name = "tsqa"
    description = (
        "Tool-augmented time-series question answering — the TimeART "
        "ablation (arXiv:2601.13653) with Aion as the toolbelt."
    )
    citation = "Wu et al., TimeART: Towards Agentic Time Series Reasoning via Tool-Augmentation, arXiv:2601.13653"
    homepage = "https://arxiv.org/abs/2601.13653"
    conditions = ("direct", "aion")

    def load(self, options: LoadOptions) -> list[BenchTask]:
        corpus = options.config or "smoke"
        if corpus not in CORPUS_PROFILES:
            raise BenchError(
                "UNKNOWN_TSQA_CORPUS",
                f"Unknown TSQA corpus {corpus!r}.",
                {"supported": sorted(CORPUS_PROFILES)},
            )
        profile = CORPUS_PROFILES[corpus]
        field_map = {**profile.get("field_map", DEFAULT_FIELD_MAP), **dict(options.extra.get("field_map") or {})}

        path = options.data_path or profile.get("path")
        if not path:
            repo = options.extra.get("dataset_repo")
            data_file = options.extra.get("dataset_file")
            if not repo or not data_file:
                raise BenchError(
                    "TSQA_CORPUS_NOT_LOCATED",
                    f"Corpus {corpus!r} needs a source. Pass --data-path for a local "
                    "file, or --dataset-repo/--dataset-file for a Hugging Face dataset. "
                    "Run the built-in `smoke` corpus to exercise the suite offline.",
                    {"corpus": corpus},
                )
            path = fetch(hf_url(str(repo), str(data_file), revision=options.revision), cache=options.cache)

        rows = read_table(path)
        tasks = [
            task for task in
            (self._to_task(index, row, field_map) for index, row in enumerate(rows))
            if task
        ]
        if not tasks:
            raise BenchError(
                "TSQA_NO_TASKS",
                f"No usable rows in {path}. Check the field map.",
                {"field_map": field_map, "path": str(path)},
            )
        return self.subsample(tasks, options)

    def _to_task(self, index: int, row: Mapping[str, Any], field_map: Mapping[str, str]) -> BenchTask | None:
        question = str(row.get(field_map["question"]) or "").strip()
        choices = _as_options(row.get(field_map["options"]))
        answer_index = _answer_index(row.get(field_map["answer"]), choices)
        if not question or len(choices) < 2 or answer_index is None:
            return None

        series = _as_series(row.get(field_map["series"]))
        context = str(row.get(field_map.get("context", "context")) or "").strip()
        prompt = f"{question}\n\nContext: {context}" if context else question
        raw_id = row.get(field_map["id"], index)

        return BenchTask(
            task_id=f"tsqa-{raw_id}",
            suite=self.name,
            kind="multiple_choice",
            prompt=prompt,
            series=series,
            options=tuple(choices),
            answer=LETTERS[answer_index],
            metric="accuracy",
            # n-way is reported separately: TimeART's 3-way and 5-way numbers
            # are not comparable to each other and must not be pooled.
            category=f"{len(choices)}-way",
            subcategory=str(row.get(field_map.get("category", "category")) or "").strip(),
            metadata={"n_way": len(choices), "has_series": bool(series)},
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
            outcome.detail = "No option letter in the response."
            return outcome

        outcome.answered = LETTERS[chosen]
        outcome.correct = outcome.answered == task.answer
        outcome.score = 1.0 if outcome.correct else 0.0
        if not outcome.correct:
            outcome.failure = "threshold"
            outcome.detail = f"Chose {outcome.answered}, expected {task.answer}."
        elif attempt.condition.startswith("aion") and not attempt.grounded:
            # Right answer, no working. Worth surfacing: a treatment arm whose
            # wins never touched a tool is measuring the prompt, not the tools.
            outcome.detail = "Correct, but no Aion tool call succeeded for this item."
        return outcome


# -- row parsing -----------------------------------------------------------

def _as_options(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            import ast

            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []
            return [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else []
        return []
    if isinstance(value, Mapping):
        return [str(value[key]) for key in sorted(value)]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def _answer_index(value: Any, options: Sequence[str]) -> int | None:
    if value is None or isinstance(value, bool) or not options:
        return None
    if isinstance(value, int):
        return value if 0 <= value < len(options) else None

    text = str(value).strip()
    stripped = text.strip("().: \t'\"")
    if len(stripped) == 1 and stripped.upper() in LETTERS[:len(options)]:
        return LETTERS.index(stripped.upper())
    if stripped.isdigit():
        index = int(stripped)
        if 0 <= index < len(options):
            return index
        if 1 <= index <= len(options):
            return index - 1
        return None
    normalised = " ".join(text.lower().split())
    for index, option in enumerate(options):
        if " ".join(str(option).lower().split()) == normalised:
            return index
    return None


def _as_series(value: Any) -> tuple[SeriesRef, ...]:
    """Accept a bare array, a list of arrays, or a list of named objects."""
    if value is None:
        return ()
    if isinstance(value, str):
        import json

        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ()

    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, Sequence):
        return ()

    numbers = _numbers(value)
    if numbers:
        return (SeriesRef(name="series", values=tuple(numbers)),)

    series: list[SeriesRef] = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            values = _numbers(item.get("values") or item.get("data") or [])
            name = str(item.get("name") or f"series_{index + 1}")
            frequency = str(item.get("frequency") or "D")
        else:
            values = _numbers(item)
            name = f"series_{index + 1}"
            frequency = "D"
        if len(values) >= 2:
            series.append(SeriesRef(name=name, values=tuple(values), frequency=frequency))
    return tuple(series)


def _numbers(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return []
        out.append(float(item))
    return out
