"""Bridge to TSAIA's own evaluators.

We do not reimplement TSAIA's grading. We fetch ``gen_eval.py`` from
USC-Melady/TSAIA, pinned to a commit, and call their evaluator classes
directly. A paraphrase would be a fidelity bug waiting to happen — their
criteria are specific (a MAPE bar that arrives *per question* in the
constraint dict, point-adjusted F1, a cumulative-return backtest, strict
equality for causal tasks) and each of the thirteen task types differs.

Why fetch instead of vendor: the TSAIA repository ships **no LICENSE file**.
The dataset is CC-BY-4.0; the code carries no grant, which under default
copyright means we may not redistribute it. Fetching it into the user's own
cache at run time is both legally clean and more faithful — the bytes that
grade a run are theirs, and the pinned SHA is recorded in the manifest.

The pin matters for a second reason: their ``main`` can move. An unpinned
harness would silently re-grade old runs against new criteria.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

from .contracts import BenchError
from .datasets import cache_dir, fetch

TSAIA_REPO = "USC-Melady/TSAIA"

# Pinned deliberately. Bump only alongside a re-validation run — changing
# this changes every score the suite produces.
TSAIA_COMMIT = "a1f7173a15f65c704dbeb962c55ecb7087833c1f"
TSAIA_GEN_EVAL_SHA256 = "2709e1a93ebd4ef8419f22aed7a5e09448407137820fff835244be9e7986a739"

_RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"

# From static_query_codeact.py. Selects which key of the evaluator's result
# dict is the reported inference-quality metric for a task type.
METRIC_NAME_MAP: dict[str, Any] = {
    "easy_stock": {"future price": "mape", "future volatility": "mape", "future trend": "accuracy"},
    "stock": "total_return",
    "electricity_prediction": "mape",
    "electricity_prediction_single": "mape",
    "causal_relation": "accuracy",
    "causal_knowledge": "accuracy",
    "climate_anomaly": "f1",
    "climate_anomaly_large": "f1",
    "energy_anomaly": "f1",
    "ecg_anomaly": "f1",
    "electricity_prediction_large": "mape",
    "stock_investment": "result",
    "stock_ir_estimation": "absolute_diff",
    "stock_rv_estimation": "absolute_diff",
    "stock_var_estimation": "absolute_diff",
}

_module_lock = threading.Lock()
_module: Any = None


def load_gen_eval(cache: str | Path | None = None) -> Any:
    """Fetch (once) and import TSAIA's ``gen_eval`` module.

    Cached per process. Import is guarded by a lock because the runner is
    threaded and importing the same module twice concurrently would race.
    """
    global _module
    if _module is not None:
        return _module

    with _module_lock:
        if _module is not None:
            return _module

        url = _RAW.format(repo=TSAIA_REPO, commit=TSAIA_COMMIT, path="gen_eval.py")
        try:
            path = fetch(url, cache=cache, filename=f"tsaia-gen_eval-{TSAIA_COMMIT[:12]}.py")
        except BenchError as exc:
            raise BenchError(
                "TSAIA_EVALUATOR_UNAVAILABLE",
                "Could not fetch TSAIA's evaluator, which this suite grades with. "
                f"Download {url} manually into the cache to run offline.",
                {"url": url, "cause": exc.message},
            ) from exc

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != TSAIA_GEN_EVAL_SHA256:
            raise BenchError(
                "TSAIA_EVALUATOR_CHANGED",
                "TSAIA's gen_eval.py does not match the pinned checksum. Refusing to "
                "grade: scores from a different evaluator are not comparable to a "
                "run that used the pinned one.",
                {"expected": TSAIA_GEN_EVAL_SHA256, "actual": digest, "path": str(path)},
            )

        try:
            spec = importlib.util.spec_from_file_location("aionbench_tsaia_gen_eval", path)
            module = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
            sys.modules["aionbench_tsaia_gen_eval"] = module
            spec.loader.exec_module(module)                          # type: ignore[union-attr]
        except ImportError as exc:
            raise BenchError(
                "TSAIA_EVALUATOR_DEPENDENCIES_MISSING",
                "TSAIA's evaluator needs numpy, pandas, scikit-learn and joblib. "
                "Install the bench extra: pip install -e '.[bench]'",
                {"cause": str(exc)},
            ) from exc

        _module = module
        return module


def evaluator_for(question_type: str, cache: str | Path | None = None) -> type:
    """Their evaluator class for a question type, via ``Question_Type_MAP``."""
    module = load_gen_eval(cache)
    base = question_type.split("-")[0]
    entry = module.Question_Type_MAP.get(base)
    if entry is None:
        raise BenchError(
            "TSAIA_UNKNOWN_QUESTION_TYPE",
            f"TSAIA's registry has no evaluator for {base!r}.",
            {"question_type": question_type,
             "known": sorted(module.Question_Type_MAP)},
        )
    return entry[1]


def metric_name_for(question_type: str) -> str | None:
    """The sub-metric key this task type reports, or None if untabulated."""
    base, _, sub_case = question_type.partition("-")
    entry = METRIC_NAME_MAP.get(base)
    if isinstance(entry, Mapping):
        return entry.get(sub_case)
    return entry


def is_problematic(question_type: str, ground_truth: Any, executor_variables: Mapping[str, Any]) -> bool:
    """TSAIA's ``Querier.check_problematic``, reproduced.

    These questions are skipped before the model ever sees them — but they
    stay in the denominator of the headline success rate, so dropping them
    silently would inflate every score.
    """
    import numpy as np

    base = question_type.split("-")[0]
    try:
        if base == "electricity_prediction" and np.min(ground_truth) < 1e-4:
            return True
        if base == "stock":
            values = executor_variables.get("VAL")
            if values is not None and np.any(np.isnan(values).values):
                return True
        if np.any(np.isnan(ground_truth)):
            return True
    except (TypeError, ValueError, AttributeError):
        # Non-numeric ground truth (causal matrices, dicts) cannot be NaN-checked
        # the way their code assumes; those types are never problematic upstream.
        return False
    return False


def evaluate(
    question_type: str,
    *,
    response: Any,
    ground_truth: Any,
    context: Any,
    constraint: Any,
    cache: str | Path | None = None,
) -> dict[str, Any]:
    """Grade one answer with TSAIA's evaluator.

    Returns their raw result dict. ``status`` is 1 or 0 and is the only
    field that decides success; the rest carries the inference-quality
    metrics. An evaluator that raises scores 0, exactly as in their
    ``Querier.query`` loop.
    """
    evaluator_class = evaluator_for(question_type, cache)
    try:
        evaluator = evaluator_class(
            response=response, ground_truth_data=ground_truth,
            context=context, constraint=constraint,
        )
        result = evaluator.evaluate()
    except Exception as exc:  # noqa: BLE001 - matches their bare except
        return {"status": 0, "error": f"{type(exc).__name__}: {exc}"}

    if not isinstance(result, Mapping):
        return {"status": 0, "error": f"Evaluator returned {type(result).__name__}, not a dict."}
    return dict(result)


def describe() -> dict[str, Any]:
    """Evaluator provenance for the run manifest."""
    return {
        "source": f"{TSAIA_REPO}@{TSAIA_COMMIT}",
        "file": "gen_eval.py",
        "sha256": TSAIA_GEN_EVAL_SHA256,
        "fetched_not_vendored": True,
        "reason": "TSAIA ships no LICENSE file; its code carries no redistribution grant.",
    }
