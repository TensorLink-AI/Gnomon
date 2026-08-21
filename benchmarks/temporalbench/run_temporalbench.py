"""Run TemporalBench (arXiv:2602.13272) conditions and score them with
the official metric module shipped in the dataset.

Conditions
----------
``control``     the row's official prompt sent verbatim to an OpenRouter
                model; its JSON answer is scored as-is.
``gnomon-pure``   T2/T4 only, no LLM: Gnomon forecasts every target channel;
                multiple-choice answers are 'Uncertain' (an honest
                abstention the option sets allow).
``gnomon-agent``  Gnomon computes the evidence (per-channel forecasts,
                season, anomalies, stats); the LLM sees the official
                prompt plus that evidence and answers only the choice
                questions. Forecast arrays in the final answer are the
                Gnomon arrays — the model cannot edit them.
``gnomon-mcp``    the tool-use arm, on every tier. The model holds the real
                ``gnomon mcp serve`` tool surface verbatim (see
                ``mcp_agent.py``) and drives the engine itself. On T2/T4
                it submits, per channel, a Gnomon artifact (used
                verbatim), its own values (labeled ``model``), or an
                abstention — the route is recorded per channel. On T1/T3
                there is nothing to forecast, so the same surface is
                offered with the tier's own answer shape: whether tool
                access helps a question-answering tier is measured, not
                assumed either way by pruning the tools to fit.

Success semantics
-----------------
On T2/T4 the per-record ``success`` boolean records COMPLETION — the
official metric module returned metrics for the row — not forecast
accuracy, so a success-rate uplift on those tiers measures completion
rate only. Forecast accuracy lives in the per-row SMAPE under each
record's ``extra`` and in the summary's scored-rows-only metric means;
compare arms with ``benchmarks/report.py``'s matched join.

Examples
--------
::

    python -m benchmarks.temporalbench.run_temporalbench --download \
        --data-dir ~/temporalbench

    python -m benchmarks.temporalbench.run_temporalbench \
        --data-dir ~/temporalbench --condition control \
        --model openai/gpt-4o --tiers T2,T4 --limit 50 \
        --output-dir results/tb-control

    python -m benchmarks.temporalbench.run_temporalbench \
        --data-dir ~/temporalbench --condition gnomon-agent \
        --model openai/gpt-4o --tiers T2,T4 --limit 50 \
        --output-dir results/tb-gnomon

    gnomon eval compare --baseline results/tb-control/gnomonbench.jsonl \
                      --treatment results/tb-gnomon/gnomonbench.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.common.manifest import write_manifest  # noqa: E402
from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402
from benchmarks.common.records import RecordWriter, RunRecord  # noqa: E402
from benchmarks.temporalbench import gnomon_runner, scoring  # noqa: E402
from benchmarks.temporalbench.tasks import (  # noqa: E402
    TIERS,
    download,
    extract_json_object,
    iter_rows,
    load_official_metrics,
)

AGENT_PREAMBLE = """\
Deterministic tool evidence computed from the task's own data by the
Gnomon engine (backtested forecasts, graded anomaly detection, season
detection). Base every numeric judgement on this evidence. On T2/T4,
return only the task's `mcq` object and omit `forecast`: the harness owns and
injects Gnomon's immutable arrays after your answer. Repeating hundreds of
forecast values here can only introduce truncation or transcription error.

<gnomon_evidence>
{evidence}
</gnomon_evidence>

"""

EVIDENCE_BUDGET = 40_000


def infrastructure_failure(error: Exception) -> bool:
    """Whether replaying the whole row is safe and potentially useful.

    The HTTP client already retries individual requests.  TemporalBench rows
    are multi-request conversations, though, and a provider outage on the
    final submit previously discarded all earlier work and permanently shrank
    the score denominator.  Retrying the row is safe: benchmark tasks are
    immutable and every MCP run gets a fresh jailed workspace.
    """
    name = type(error).__name__
    detail = str(error).lower()
    if name in {"OpenRouterError", "IncompleteRead", "RemoteDisconnected",
                "TimeoutError", "ConnectionError"}:
        return True
    return (
        "mcp tools/list failed" in detail
        or "connection reset" in detail
        or "temporarily unavailable" in detail
    )


def bounded_evidence(digest: dict[str, Any],
                     budget: int = EVIDENCE_BUDGET) -> str:
    """Serialize the evidence digest to at most ``budget`` characters of
    VALID JSON. A plain ``json.dumps(digest)[:budget]`` could cut the
    text mid-token and hand the agent malformed evidence, so instead the
    largest entries are progressively shrunk — long forecast arrays keep
    a prefix plus summary stats under a ``"truncated": true`` marker —
    and, if that is still not enough, top-level entries are dropped
    largest-first with the dropped keys named in the result.
    Deterministic: the same digest always yields the same string.
    Non-finite numbers become null so the output is always spec-valid
    JSON; the budget is best-effort — the residual skeleton (the
    ``truncated``/``dropped`` markers) can exceed a budget smaller than
    itself, far below any real configuration.
    """
    # Private deep copy; parse_constant turns NaN/Infinity into null so
    # the output stays spec-valid JSON whatever the stats contain.
    digest = json.loads(json.dumps(digest, default=str),
                        parse_constant=lambda _: None)
    text = json.dumps(digest)
    if len(text) <= budget:
        return text
    keep = 24  # leading forecast values kept verbatim when shrinking
    forecasts = digest.get("forecasts")
    if isinstance(forecasts, dict):
        by_size = sorted(
            forecasts, key=lambda k: (-len(json.dumps(forecasts[k])), k)
        )
        for key in by_size:
            entry = forecasts[key]
            values = entry.get("values") if isinstance(entry, dict) else None
            if not (isinstance(values, list) and len(values) > keep):
                continue
            numeric = [v for v in values if isinstance(v, (int, float))]
            entry["values"] = values[:keep]
            entry["truncated"] = True
            entry["values_total"] = len(values)
            if numeric:
                entry["values_stats"] = {
                    "min": min(numeric), "max": max(numeric),
                    "mean": round(sum(numeric) / len(numeric), 6),
                }
            text = json.dumps(digest)
            if len(text) <= budget:
                return text
    dropped: list[str] = []
    digest["truncated"] = True
    digest["dropped"] = dropped
    droppable = sorted(
        (k for k in digest if k not in ("truncated", "dropped")),
        key=lambda k: (-len(json.dumps(digest[k])), k),
    )
    for key in droppable:
        del digest[key]
        dropped.append(key)
        text = json.dumps(digest)
        if len(text) <= budget:
            return text
    return json.dumps(digest)


def answer_row(row: dict[str, Any], condition: str,
               client: OpenRouterClient | None,
               best_effort: bool = False,
               mcp_profile: str = "full",
               compile_context: bool = False,
               context_receipts_dir: str | None = None,
               compile_questions: bool = False,
               question_receipts_dir: str | None = None,
               mcp_call_timeout: float | None = None,
               named_tsfm: str | None = None,
               model_evidence_registry: str | None = None) -> dict[str, Any]:
    """Produce the row's answer object under the given condition.

    ``channel_support`` in the result maps each forecast channel to its
    Gnomon support label; with ``best_effort`` enabled it is how a
    disclosed-fallback channel stays distinguishable from a supported
    one all the way into the details records and the summary.
    """
    if condition == "control":
        messages = [{"role": "user", "content": row["prompt"]}]
        completion = client.completions(messages, n=1)[0]
        try:
            answer = extract_json_object(completion)
        except ValueError:
            # A prose-only response is a transport-format failure, not task
            # accuracy. Give the untooled baseline one bounded repair turn,
            # just as a real host would, while retaining the extra call and
            # tokens in its economics. Do not add evidence or alter values.
            messages.extend([
                {"role": "assistant", "content": completion},
                {"role": "user", "content": (
                    "Return the same answer now as one valid JSON object only. "
                    "Do not revise, recompute, add, or remove any values.")},
            ])
            completion = client.completions(messages, n=1)[0]
            answer = extract_json_object(completion)
        return {"answer": answer, "abstained": [], "channel_support": {}}

    if condition == "gnomon-mcp":
        from benchmarks.temporalbench.mcp_agent import mcq_row, run_row

        # best_effort is deliberately not passed: on this arm the model
        # itself decides whether to request the engine's labeled
        # fallback via the gnomon_forecast tool — the realistic path.
        # T1/T3 carry no forecast channels, so they take the same
        # session with the tier's own answer shape.
        # Keep tier-independent session options separate from forecast-only
        # model-admission options. Question compilation applies to both the
        # forecasting questions and T3's descriptive question pack.
        common_args = ({} if mcp_profile == "full"
                       else {"profile": mcp_profile})
        if compile_context:
            common_args["compile_context"] = True
            if context_receipts_dir:
                common_args["context_receipts_dir"] = context_receipts_dir
        if mcp_call_timeout is not None:
            common_args["mcp_call_timeout"] = mcp_call_timeout
        if compile_questions:
            common_args["compile_questions"] = True
            if question_receipts_dir:
                common_args["question_receipts_dir"] = question_receipts_dir
        if row.get("tier") not in ("T2", "T4"):
            return mcq_row(row, client, **common_args)

        forecast_args = dict(common_args)
        if model_evidence_registry:
            forecast_args["model_evidence_registry"] = model_evidence_registry
        return run_row(row, client, **forecast_args)

    analysis = gnomon_runner.analyse_row(
        row, best_effort=best_effort, named_tsfm=named_tsfm)
    tier = row.get("tier")
    if condition == "gnomon-pure":
        if tier not in ("T2", "T4"):
            raise ValueError("gnomon-pure covers tiers T2 and T4 only")
        forecast, abstained, support = gnomon_runner.forecast_payload(analysis)
        mcq, mcq_abstained = gnomon_runner.uncertain_mcq(row)
        return {
            "answer": {"forecast": forecast, "mcq": mcq},
            "abstained": abstained + mcq_abstained, "analysis": analysis,
            "channel_support": support,
        }

    # gnomon-agent: evidence digest + official prompt; LLM answers choices.
    digest = {k: v for k, v in analysis.items() if k != "channels"}
    digest["forecasts"] = {
        key: (outcome if outcome.get("abstained") else
              {"support": outcome["support"],
               "selected_model": outcome["selected_model"],
               "values": outcome["values"]})
        for key, outcome in analysis.get("channels", {}).items()
    }
    prompt = AGENT_PREAMBLE.format(
        evidence=bounded_evidence(digest)
    ) + row["prompt"]
    completion = client.completions(
        [{"role": "user", "content": prompt}], n=1
    )[0]
    try:
        answer = extract_json_object(completion)
    except ValueError:
        # Evidence arms need the same bounded format repair as the raw
        # control. The model has already reasoned; this turn may only package
        # its existing choices, never recompute them or alter Gnomon's arrays.
        completion = client.completions([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
            {"role": "user", "content": (
                "Return only your existing multiple-choice answers as "
                "{\"mcq\": {...}} JSON. Do not explain, recompute, or include "
                "forecast arrays; the harness injects those immutably.")},
        ], n=1)[0]
        answer = extract_json_object(completion)
    abstained: list[str] = []
    support: dict[str, str] = {}
    if tier in ("T2", "T4"):
        forecast, abstained, support = gnomon_runner.forecast_payload(analysis)
        answer["forecast"] = forecast  # Gnomon owns the numbers.
    return {"answer": answer, "abstained": abstained, "analysis": analysis,
            "channel_support": support}


def score_row(row: dict[str, Any], answer: dict[str, Any],
              official_metrics) -> dict[str, Any]:
    tier = row.get("tier")
    if tier == "T1":
        return {"tier": tier, "choice": scoring.score_t1(row, answer)}
    if tier == "T3":
        return {"tier": tier,
                "choice": scoring.score_t3(row, answer.get("answers") or [])}
    metrics, flag = scoring.score_forecast(
        row, answer.get("forecast"), official_metrics
    )
    return {"tier": tier,
            "choice": scoring.score_mcq(row, answer.get("mcq") or {}),
            "forecast_metrics": metrics, "metric_flag": flag}


def align_typed_answers(
    question_keys: list[str], answers: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Align compiler IDs without guessing from labels or answer options.

    Semantic IDs bind exactly. Positional IDs q1/q2/... bind to the stable
    source-question order. Anything else remains unaligned rather than being
    credited to whichever task answer happens to be available.
    """
    aligned: dict[str, dict[str, Any]] = {}
    for answer in answers:
        raw_id = str((answer.get("question") or {}).get("id") or "")
        base_id = raw_id.split(":", 1)[0]
        key = base_id if base_id in question_keys else None
        if key is None and base_id.startswith("q") and base_id[1:].isdigit():
            index = int(base_id[1:]) - 1
            if 0 <= index < len(question_keys):
                key = question_keys[index]
        if key is not None:
            aligned[key] = answer
    return aligned


def load_resumable_records(*paths: Path) -> dict[str, dict[str, Any]]:
    """Recover complete durable lines from canonical and partial runs."""
    records: dict[str, dict[str, Any]] = {}
    for source in paths:
        if not source.is_file():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = str(record.get("task_id"))
            previous = records.get(identifier)
            if previous is not None and previous != record:
                raise ValueError(f"conflicting resumable records for {identifier}")
            records[identifier] = record
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--condition",
                        choices=["control", "gnomon-pure", "gnomon-agent",
                                 "gnomon-mcp"])
    parser.add_argument("--model", default=None, help="OpenRouter model id")
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible chat-completions endpoint to send the "
             "model's requests to (default: $OPENROUTER_BASE_URL, else "
             "OpenRouter). The resolved endpoint is recorded in "
             "summary.json's llm_usage and in the run manifest: the same "
             "model id served from elsewhere is a different measurement.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                        choices=("OPENROUTER_API_KEY", "ENGY_API_KEY", "CHUTES_API_KEY"),
                        help="Credential variable for the selected endpoint.")
    parser.add_argument("--tiers", default="T1,T2,T3,T4")
    parser.add_argument("--datasets", default=None,
                        help="Comma list of source datasets to keep")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many rows after tier/dataset filters; "
                             "supports resumable one-row benchmark shards.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--request-timeout", type=int, default=180,
                        help="seconds per provider request (default: 180)")
    parser.add_argument(
        "--max-retries", type=int, default=0,
        help=("retries inside one provider call (default: 0); TemporalBench "
              "already retries infrastructure failures at row scope, so "
              "raising both values multiplies outage latency"))
    parser.add_argument(
        "--infrastructure-retries", type=int, default=2,
        help=("whole-row retries after exhausted provider/transport or MCP "
              "startup failures; product non-submission is never retried"))
    parser.add_argument(
        "--resume", action="store_true",
        help=("reuse rows with saved answer details and execute only missing "
              "or errored rows; all rows are rescored into a fresh summary"))
    parser.add_argument(
        "--retry-voided", action="store_true",
        help=("With --resume, rerun rows previously voided by a harness cap "
              "instead of replaying their abstention record."))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--mcp-profile", choices=["full", "core", "describe", "evidence", "mega", "decision", "data"],
        default="evidence",
        help="Tool profile offered by the gnomon-mcp condition. Compare "
             "profiles only through matched runs over the same rows, model, "
             "prompt, temperature, endpoint, and harness caps.")
    parser.add_argument(
        "--compile-context", action="store_true",
        help="Gnomon MCP only: run Gnomon's quoted text-to-context compiler "
             "for T3/T4, inject only deterministically validated events into "
             "forecast/run, and expose the validation receipt to the agent.")
    parser.add_argument(
        "--context-receipts-dir",
        help="With --compile-context, persist one validated compiler receipt "
             "per task and replay it on later runs. Use the same directory "
             "for every surface in a matched experiment so compiler "
             "randomness is not attributed to the tool surface.")
    parser.add_argument(
        "--compile-questions", action="store_true",
        help="Gnomon MCP T2/T4 only: compile question text into validated typed intent before execution.")
    parser.add_argument(
        "--question-receipts-dir",
        help="Persist immutable typed-intent compiler receipts for matched replay.")
    parser.add_argument(
        "--best-effort", action="store_true",
        help="Gnomon conditions only: enable the engine's disclosed "
             "best-effort fallback on channels that would abstain "
             "(sparse channels like MIMIC's temperature_c). Fallback "
             "rows carry support 'best_effort' and a NO RELIABLE "
             "FORECAST warning — they are not supported forecasts, and "
             "the summary reports the support-label mix beside every "
             "score that includes them. Default off: an abstention "
             "stays an abstention unless explicitly traded for a "
             "labeled fallback.")
    parser.add_argument(
        "--named-tsfm",
        help="gnomon-agent/gnomon-pure model-supply experiment: use this "
             "pinned sandbox TSFM directly and label its forecasts "
             "experimental_named_model. This bypasses local candidate "
             "selection and is not Gnomon's governed default.")
    parser.add_argument(
        "--model-evidence-registry",
        help="gnomon-mcp: copy this versioned registry into each jailed row "
             "and explicitly request evidence-weighted model admission.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.download:
        download(data_dir)
        print(f"Dataset ready under {data_dir}")
        if not args.condition:
            return 0
    if not args.condition or not args.output_dir:
        parser.error("--condition and --output-dir are required to run")
    if args.condition != "gnomon-pure" and not args.model:
        parser.error("--model is required for this condition")
    if args.best_effort and args.condition == "control":
        parser.error("--best-effort applies to the Gnomon conditions only")
    if args.best_effort and args.condition == "gnomon-mcp":
        parser.error("--best-effort does not apply to gnomon-mcp: the model "
                     "decides itself whether to request the engine's "
                     "labeled fallback via the gnomon_forecast tool")
    if args.compile_context and args.condition != "gnomon-mcp":
        parser.error("--compile-context applies to gnomon-mcp only")
    if args.context_receipts_dir and not args.compile_context:
        parser.error("--context-receipts-dir requires --compile-context")
    if args.compile_questions and args.condition != "gnomon-mcp":
        parser.error("--compile-questions applies to gnomon-mcp only")
    if args.question_receipts_dir and not args.compile_questions:
        parser.error("--question-receipts-dir requires --compile-questions")
    if args.named_tsfm and args.condition not in {"gnomon-agent", "gnomon-pure"}:
        parser.error("--named-tsfm applies only to gnomon-agent or gnomon-pure")

    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip() in TIERS)
    # gnomon-pure produces forecasts and nothing else, so it is a T2/T4
    # condition by construction. gnomon-mcp is not: the tool surface is
    # offered on every tier, and whether an agent that can interrogate a
    # series answers T1/T3 better is a measurement, not an assumption.
    if args.condition == "gnomon-pure":
        tiers = tuple(t for t in tiers if t in ("T2", "T4")) or ("T2", "T4")
    datasets = (tuple(d.strip() for d in args.datasets.split(","))
                if args.datasets else None)
    official_metrics = load_official_metrics(data_dir)
    from benchmarks.common.envfile import load_env_file
    load_env_file()
    client = (OpenRouterClient(args.model, api_key=os.environ.get(args.api_key_env),
                               temperature=args.temperature,
                               max_tokens=8000, base_url=args.base_url,
                               timeout=args.request_timeout,
                               max_retries=args.max_retries)
              if args.model else None)

    output_dir = Path(args.output_dir)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "gnomonbench.jsonl"
    partial_records_path = output_dir / "gnomonbench.partial.jsonl"
    prior_summary_path = output_dir / "summary.json"
    prior_summary = (json.loads(prior_summary_path.read_text())
                     if args.resume and prior_summary_path.is_file() else {})
    prior_records = (load_resumable_records(records_path, partial_records_path)
                     if args.resume else {})
    # Write a complete replacement beside the canonical record file and
    # publish it atomically only after every selected row has finished.
    # A provider, MCP child, or operator interrupt must not destroy the
    # prior resumable run halfway through its rewrite.
    partial_records_path.unlink(missing_ok=True)
    partial_records_path.touch()
    records = RecordWriter(partial_records_path)

    choice_by_tier: dict[str, list[int]] = {}
    choice_rows_by_tier: dict[str, int] = {}
    forecast_metrics_acc: dict[str, list[float]] = {}
    sensitivity_metrics_acc: dict[str, list[float]] = {}
    sensitivity_smape_deltas: list[float] = []
    sensitivity_rows = sensitivity_channels = 0
    sensitivity_wins = sensitivity_losses = sensitivity_ties = 0
    abstained_rows = errored = forecast_rows_scored = rows_voided = 0
    # Coverage beside every figure: how many channel forecasts each
    # number rests on, by support label, plus the abstained channels.
    support_mix: dict[str, int] = {}
    # gnomon-mcp: per-channel routes (gnomon / informed-direct / direct
    # / abstain) so the exits stay separable in analysis.
    route_mix: dict[str, int] = {}
    mcp_calls_seen: list[int] = []
    mcp_run_tokens = 0
    mcp_schema_bytes: set[int] = set()
    mcp_rows_answered = 0
    compiler_calls = compiler_events = compiler_hypotheses = compiler_rejected = 0
    compiler_receipts_reused = 0
    question_compiler_calls = question_compiler_accepted = 0
    question_compiler_rejected = question_compiler_receipts_reused = 0
    temporal_answer_receipts = temporal_answers_returned = 0
    temporal_primary_unchanged = 0
    typed_questions_requested = typed_questions_with_engine_answer = 0
    typed_engine_answers_officially_comparable = 0
    typed_engine_answers_officially_correct = 0
    typed_answers_comparable_to_submission = 0
    typed_answers_preserved_by_agent = 0
    canonical_choice_correct = canonical_choice_total = 0
    synthesized_choice_correct = synthesized_choice_total = 0
    hybrid_choice_correct = hybrid_choice_total = 0
    advisory_overrides = advisory_overrides_helped = advisory_overrides_hurt = 0
    context_channels_considered = context_events_admitted = 0
    context_events_rejected = context_events_applied = 0
    context_events_scenario_only = 0
    covariate_channels_considered = covariate_channels_admitted = 0
    infrastructure_retries = 0
    infrastructure_failures: dict[str, int] = {}
    terminal_errors: dict[str, int] = {}
    resumed_rows = 0
    channels_abstained = 0
    total = 0
    # Denominator for the scored-only forecast means: every T2/T4 row the
    # run saw, so the coverage behind those means is a ratio in the
    # summary, not an inference from three separate counters.
    forecast_rows_total = 0
    requested = ((args.offset + args.limit) if args.limit is not None else None)
    for row_index, row in enumerate(iter_rows(
            data_dir, tiers=tiers or TIERS, datasets=datasets, limit=requested)):
        if row_index < args.offset:
            continue
        total += 1
        if row.get("tier") in ("T2", "T4"):
            forecast_rows_total += 1
        row_id = row.get("id", f"row{total}")
        started = time.time()
        retained_latency: float | None = None
        # Answering and scoring fail separately: a scoring exception must
        # not swallow an answer that abstained — the abstention flag has
        # to survive into the record either way.
        try:
            detail_path = details_dir / f"{row_id}.json"
            prior_record = prior_records.get(str(row_id))
            if (args.resume and detail_path.is_file() and prior_record
                    and not prior_record.get("error")
                    and not (args.retry_voided
                             and prior_record.get("row_abstained"))):
                saved = json.loads(detail_path.read_text())
                outcome = {
                    "answer": saved.get("answer") or {},
                    "abstained": saved.get("abstained") or [],
                    "channel_support": saved.get("channel_support") or {},
                    "channel_route": saved.get("channel_route") or {},
                    "context_execution": saved.get("context_execution") or {},
                    "covariate_execution": saved.get("covariate_execution") or {},
                    "sensitivity_forecast": saved.get(
                        "sensitivity_forecast") or {},
                    "canonical_mcq": saved.get("canonical_mcq") or {},
                    "synthesized_mcq": saved.get("synthesized_mcq") or {},
                    "choice_authority": saved.get("choice_authority") or {},
                    "choice_basis": saved.get("choice_basis") or {},
                    "mcp": saved.get("mcp") or {},
                    **({"row_abstained": True}
                       if prior_record.get("row_abstained") else {}),
                }
                retained_latency = float(prior_record.get(
                    "latency_seconds") or 0.0)
                resumed_rows += 1
            else:
                row_retry = 0
                while True:
                    try:
                        outcome = answer_row(
                            row, args.condition, client,
                            best_effort=args.best_effort,
                            mcp_profile=args.mcp_profile,
                            compile_context=args.compile_context,
                            context_receipts_dir=args.context_receipts_dir,
                            compile_questions=args.compile_questions,
                            question_receipts_dir=args.question_receipts_dir,
                            mcp_call_timeout=args.request_timeout,
                            named_tsfm=args.named_tsfm,
                            model_evidence_registry=args.model_evidence_registry,
                        )
                        break
                    except Exception as error:
                        if (not infrastructure_failure(error)
                                or row_retry >= args.infrastructure_retries):
                            raise
                        row_retry += 1
                        infrastructure_retries += 1
                        key = type(error).__name__ + ": " + str(error)[:120]
                        infrastructure_failures[key] = (
                            infrastructure_failures.get(key, 0) + 1)
        except Exception as error:
            errored += 1
            key = type(error).__name__ + ": " + str(error)[:160]
            terminal_errors[key] = terminal_errors.get(key, 0) + 1
            records.write(RunRecord(task_id=row_id, success=False,
                                    extra={"error": str(error)[:400],
                                           "error_type": type(error).__name__,
                                           "traceback": traceback.format_exc()[-2000:],
                                           "error_stage": "answer"}))
            continue
        try:
            verdict = score_row(row, outcome["answer"], official_metrics)
        except Exception as error:
            errored += 1
            key = "score/" + type(error).__name__ + ": " + str(error)[:160]
            terminal_errors[key] = terminal_errors.get(key, 0) + 1
            records.write(RunRecord(
                task_id=row_id, success=False,
                appropriate_abstention=bool(outcome.get("abstained")),
                extra={"error": str(error)[:400], "error_stage": "score",
                       "abstained": outcome.get("abstained")},
            ))
            continue
        elapsed = (retained_latency if retained_latency is not None
                   else time.time() - started)

        choice = verdict.get("choice") or {}
        canonical_mcq = outcome.get("canonical_mcq") or {}
        synthesized_mcq = outcome.get("synthesized_mcq") or {}
        choice_authority = outcome.get("choice_authority") or {}
        canonical_score = scoring.score_mcq(row, canonical_mcq)
        synthesized_score = scoring.score_mcq(row, synthesized_mcq)
        canonical_choice_correct += canonical_score["correct"]
        canonical_choice_total += sum(key in canonical_mcq
                                      for key in (row.get("mcq") or {}))
        synthesized_choice_correct += synthesized_score["correct"]
        synthesized_choice_total += sum(key in synthesized_mcq
                                        for key in (row.get("mcq") or {}))
        for key, authority in choice_authority.items():
            if authority != "advisory_override" or key not in synthesized_mcq \
                    or key not in canonical_mcq:
                continue
            if str(synthesized_mcq[key]).strip().lower() == \
                    str(canonical_mcq[key]).strip().lower():
                continue
            advisory_overrides += 1
            expected = (((row.get("mcq") or {}).get(key) or {}).get("label"))
            synth_ok = str(synthesized_mcq[key]).strip().lower() == \
                str(expected).strip().lower()
            canonical_ok = str(canonical_mcq[key]).strip().lower() == \
                str(expected).strip().lower()
            advisory_overrides_helped += int(synth_ok and not canonical_ok)
            advisory_overrides_hurt += int(canonical_ok and not synth_ok)
        tier = verdict["tier"]
        # A row the harness voided (a breached cap, a run that never
        # submitted) did not answer the questions wrongly — it did not
        # answer them. Scoring its empty answer as zero-of-n would report
        # a harness cap as a model failure and drag the tier mean down by
        # exactly the rows the harness lost; it is counted as a voided
        # row instead, and reported as one.
        voided = outcome.get("row_abstained")
        if voided:
            rows_voided += 1
        if choice.get("total") and not voided:
            hybrid_choice_correct += int(choice["correct"])
            hybrid_choice_total += int(choice["total"])
            choice_rows_by_tier[tier] = choice_rows_by_tier.get(tier, 0) + 1
            choice_by_tier.setdefault(tier, []).extend(
                [1] * choice["correct"] + [0] * (choice["total"] - choice["correct"])
            )
        metrics = verdict.get("forecast_metrics")
        sensitivity_diagnostic = None
        sensitivity = outcome.get("sensitivity_forecast") or {}
        if metrics and sensitivity:
            primary_forecast = (outcome.get("answer") or {}).get("forecast") or {}
            overlay = dict(primary_forecast)
            overlay.update(sensitivity)
            overlay_metrics, overlay_flag = scoring.score_forecast(
                row, overlay, official_metrics)
            if overlay_metrics:
                primary_smape = (metrics.get("OW_sMAPE")
                                 if metrics.get("OW_sMAPE") is not None
                                 else metrics.get("SMAPE"))
                overlay_smape = (overlay_metrics.get("OW_sMAPE")
                                 if overlay_metrics.get("OW_sMAPE") is not None
                                 else overlay_metrics.get("SMAPE"))
                delta = (float(overlay_smape) - float(primary_smape)
                         if primary_smape is not None
                         and overlay_smape is not None else None)
                sensitivity_diagnostic = {
                    "policy": "retrospective_overlay_never_submitted",
                    "channels": sorted(sensitivity),
                    "primary_metrics": metrics,
                    "overlay_metrics": overlay_metrics,
                    "metric_flag": overlay_flag,
                    "smape_delta_overlay_minus_primary": delta,
                }
                sensitivity_rows += 1
                sensitivity_channels += len(sensitivity)
                for key, value in overlay_metrics.items():
                    if isinstance(value, (int, float)):
                        sensitivity_metrics_acc.setdefault(key, []).append(
                            float(value))
                if delta is not None:
                    sensitivity_smape_deltas.append(delta)
                    if delta < -1e-12:
                        sensitivity_wins += 1
                    elif delta > 1e-12:
                        sensitivity_losses += 1
                    else:
                        sensitivity_ties += 1
        if metrics:
            forecast_rows_scored += 1
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    forecast_metrics_acc.setdefault(key, []).append(float(value))
        if outcome.get("abstained"):
            abstained_rows += 1
        channel_support = outcome.get("channel_support") or {}
        for label in channel_support.values():
            support_mix[label] = support_mix.get(label, 0) + 1
        channel_route = outcome.get("channel_route") or {}
        for route in channel_route.values():
            route_mix[route] = route_mix.get(route, 0) + 1
        context_execution = outcome.get("context_execution") or {}
        for receipt in context_execution.values():
            context_channels_considered += 1
            context_events_admitted += int(receipt.get("admitted", 0))
            context_events_rejected += int(receipt.get("rejected", 0))
            context_events_applied += int(receipt.get("applied", 0))
            context_events_scenario_only += int(receipt.get(
                "scenario_only", 0))
        covariate_execution = outcome.get("covariate_execution") or {}
        for receipt in covariate_execution.values():
            covariate_channels_considered += int(bool(receipt.get("considered")))
            covariate_channels_admitted += int(bool(receipt.get("admitted")))
        # Forecast-channel coverage, so only the tiers that have forecast
        # channels contribute: a T1/T3 row's abstention is a row that
        # went unanswered, counted as such, not a channel Gnomon declined.
        if tier in ("T2", "T4"):
            channels_abstained += sum(
                1 for reason in outcome.get("abstained") or []
                if not reason.startswith("mcq/"))

        success = bool(metrics) if tier in ("T2", "T4") else (
            choice.get("total", 0) > 0 and choice["correct"] == choice["total"]
        )
        # `success` is not one quantity across tiers, and a pooled success
        # rate silently blends the two. The basis rides on every record so
        # report.py can split the rate by what was actually measured.
        success_basis = ("completion" if tier in ("T2", "T4")
                         else "all_choices_correct")
        # The MCP arm reports its real call count; the other Gnomon
        # conditions make exactly one engine invocation; the control makes
        # none. A constant 1 for the MCP arm made average_tool_calls a
        # constant, not a measurement.
        mcp_info = outcome.get("mcp") or {}
        if mcp_info:
            mcp_calls_seen.append(int(mcp_info.get("calls", 0)))
            mcp_run_tokens += int(mcp_info.get("run_tokens", 0))
            compiler_calls += int(mcp_info.get("compiler_calls", 0))
            compiler_receipts_reused += int(bool(
                mcp_info.get("context_receipt_reused")))
            temporal_compilation = mcp_info.get("temporal_compilation") or {}
            question_compiler_calls += int(bool(
                temporal_compilation.get("compiler_called")))
            question_compiler_receipts_reused += int(bool(
                temporal_compilation.get("receipt_reused")))
            question_compiler_accepted += int(
                temporal_compilation.get("accepted", 0))
            question_compiler_rejected += int(
                temporal_compilation.get("rejected", 0))
            receipt_answers: dict[str, dict[str, Any]] = {}
            for answer_receipt in mcp_info.get("temporal_answer_receipts") or []:
                temporal_answer_receipts += 1
                temporal_answers_returned += len(
                    answer_receipt.get("answers") or [])
                temporal_primary_unchanged += int(
                    answer_receipt.get("primary_forecast_unchanged") is True)
                for typed_answer in answer_receipt.get("answers") or []:
                    question = typed_answer.get("question") or {}
                    question_id = str(question.get("id") or "").split(":", 1)[0]
                    if question_id:
                        # Later receipts for the same immutable artifact do
                        # not multiply the evaluation denominator.
                        receipt_answers[question_id] = typed_answer
            if tier == "T3":
                # T3's persisted MCP record deliberately carries compiler
                # counts, not raw benchmark question text. Inline describe
                # receipts can therefore establish unique returned answers
                # and coverage, but not a label/options projection.
                requested_count = int(temporal_compilation.get("accepted", 0))
                typed_questions_requested += requested_count
                typed_questions_with_engine_answer += min(
                    requested_count, len(receipt_answers))
                requested_order = []
            else:
                requested_order = list((row.get("mcq") or {}).keys())
            requested_keys = set(requested_order)
            row_engine_answers = align_typed_answers(
                requested_order, list(receipt_answers.values()))
            typed_questions_requested += len(requested_order)
            final_choices = (outcome.get("answer") or {}).get("mcq") or {}
            per_question = choice.get("per_question") or {}
            for question_id in requested_keys & set(row_engine_answers):
                typed_questions_with_engine_answer += 1
                best = row_engine_answers[question_id].get("best_estimate") or {}
                candidates = {str(value).strip().lower() for value in (
                    best.get("value"), best.get("display_value"))
                    if value is not None}
                # The host is allowed to perform only the same deterministic
                # unambiguous vocabulary projection used at submission. Count
                # that as preservation, not as an LLM paraphrase.
                from gnomon.temporal_vocabulary import project_temporal_choice
                options = (((row.get("mcq") or {}).get(question_id) or {})
                           .get("options") or []) if tier != "T3" else []
                projected = project_temporal_choice(best.get("value"), options)
                if projected:
                    typed_engine_answers_officially_comparable += 1
                    candidates.add(str(projected["display_value"]).strip().lower())
                    expected = (((row.get("mcq") or {}).get(question_id) or {})
                                .get("label"))
                    typed_engine_answers_officially_correct += int(
                        str(projected["display_value"]).strip().lower()
                        == str(expected).strip().lower())
                if candidates:
                    typed_answers_comparable_to_submission += 1
                    typed_answers_preserved_by_agent += int(
                        str(final_choices.get(question_id, "")).strip().lower()
                        in candidates)
            compiled = mcp_info.get("compiled_context") or {}
            compiler_events += int(compiled.get("accepted_events", 0))
            compiler_hypotheses += int(compiled.get("accepted_hypotheses", 0))
            compiler_rejected += int(compiled.get("rejected", 0))
            if not outcome.get("row_abstained"):
                mcp_rows_answered += 1
            if mcp_info.get("schema_bytes") is not None:
                mcp_schema_bytes.add(int(mcp_info["schema_bytes"]))
        records.write(RunRecord(
            task_id=row_id, success=success,
            appropriate_abstention=bool(outcome.get("abstained")),
            tool_calls=(int(mcp_info["calls"]) if "calls" in mcp_info
                        else 0 if args.condition == "control" else 1),
            latency_seconds=round(elapsed, 3),
            extra={"tier": tier,
                   "success_basis": success_basis,
                   "choice_correct": None if voided else choice.get("correct"),
                   "choice_total": None if voided else choice.get("total"),
                   **({"row_abstained": voided} if voided else {}),
                   "smape": (metrics or {}).get("SMAPE")
                   or (metrics or {}).get("OW_sMAPE"),
                   **({"channel_support": channel_support}
                      if channel_support else {}),
                   **({"channel_route": channel_route}
                      if channel_route else {}),
                   **({"sensitivity_smape_delta": sensitivity_diagnostic[
                       "smape_delta_overlay_minus_primary"]}
                      if sensitivity_diagnostic else {})},
        ))
        (details_dir / f"{row_id}.json").write_text(
            json.dumps({"verdict": verdict,
                        "abstained": outcome.get("abstained"),
                        # Support labels beside the arrays: a
                        # best_effort channel is a disclosed fallback,
                        # and score_per_channel reports the mix.
                        "channel_support": channel_support or None,
                        # gnomon-mcp only: which exit each channel took
                        # and what the model did with the tools.
                        "channel_route": channel_route or None,
                        # Distinct from compiled_context: this is the numeric
                        # engine's gate and application receipt.
                        "context_execution": context_execution or None,
                        "covariate_execution": covariate_execution or None,
                        "sensitivity_forecast": sensitivity or None,
                        "sensitivity_diagnostic": sensitivity_diagnostic,
                        "canonical_mcq": canonical_mcq or None,
                        "synthesized_mcq": synthesized_mcq or None,
                        "choice_authority": choice_authority or None,
                        "choice_basis": outcome.get("choice_basis") or None,
                        "mcp": outcome.get("mcp"),
                        "answer": outcome["answer"]}, indent=2,
                       default=str) + "\n",
            encoding="utf-8",
        )
        if client is not None:
            # Provider usage must survive Ctrl-C just like completed rows.
            # Write a replaceable snapshot after each durable detail rather
            # than reconstructing token counts from compacted transcripts.
            checkpoint = output_dir / "usage.checkpoint.json"
            checkpoint_tmp = output_dir / "usage.checkpoint.tmp.json"
            checkpoint_tmp.write_text(json.dumps({
                "llm_usage": client.usage_summary,
                "completed_details": len(list(details_dir.glob("*.json"))),
            }, indent=2) + "\n", encoding="utf-8")
            os.replace(checkpoint_tmp, checkpoint)
        if total % 20 == 0:
            print(f"...{total} rows")

    os.replace(partial_records_path, records_path)
    summary = {
        "benchmark": "temporalbench",
        "condition": args.condition,
        "model": args.model,
        "tiers": list(tiers or TIERS),
        "datasets": list(datasets) if datasets else "all",
        "rows": total,
        "row_offset": args.offset,
        "rows_errored": errored,
        "run_status": ("failed" if total and errored == total else
                       "partial" if errored else "complete"),
        "terminal_error_breakdown": dict(sorted(terminal_errors.items())),
        "rows_with_abstentions": abstained_rows,
        # Rows the harness ended without an answer (a breached cap, a run
        # that never submitted). They are in `rows`, out of every
        # accuracy denominator, and quoted here so the coverage behind
        # the tier means is visible rather than inferred.
        "rows_voided_by_harness": rows_voided,
        "infrastructure_retries": infrastructure_retries,
        "infrastructure_failures_retried": dict(sorted(
            infrastructure_failures.items())),
        "resumed_rows": resumed_rows,
        "choice_accuracy_by_tier_scored_only": {
            tier: sum(flags) / len(flags)
            for tier, flags in sorted(choice_by_tier.items())
        },
        "choice_rows_scored_by_tier": dict(sorted(choice_rows_by_tier.items())),
        "choice_contract": {
            "canonical_correct": canonical_choice_correct,
            "canonical_total": canonical_choice_total,
            "canonical_accuracy": (canonical_choice_correct
                                   / canonical_choice_total
                                   if canonical_choice_total else None),
            "synthesized_correct": synthesized_choice_correct,
            "synthesized_total": synthesized_choice_total,
            "synthesized_accuracy": (synthesized_choice_correct
                                     / synthesized_choice_total
                                     if synthesized_choice_total else None),
            "hybrid_correct": hybrid_choice_correct,
            "hybrid_total": hybrid_choice_total,
            "hybrid_accuracy": (hybrid_choice_correct / hybrid_choice_total
                                if hybrid_choice_total else None),
            "advisory_overrides": advisory_overrides,
            "advisory_overrides_helped": advisory_overrides_helped,
            "advisory_overrides_hurt": advisory_overrides_hurt,
            "primary_forecast_unchanged": True,
        },
        "forecast_metrics_mean_scored_only": {
            key: sum(values) / len(values)
            for key, values in sorted(forecast_metrics_acc.items())
            if values and key in ("MAPE", "MAE", "RMSE", "SMAPE",
                                  "OW_sMAPE", "OW_RMSSE", "OW_MASE")
        },
        "forecast_rows_scored": forecast_rows_scored,
        "forecast_rows_total": forecast_rows_total,
        # The coverage behind the scored-only means, as a ratio: a mean
        # over 12% of the rows and a mean over 96% of them print the same
        # number of digits.
        "forecast_scored_fraction": (
            round(forecast_rows_scored / forecast_rows_total, 4)
            if forecast_rows_total else None
        ),
        "sensitivity_diagnostic": {
            "policy": "retrospective_overlay_never_submitted",
            "rows_with_scenario": sensitivity_rows,
            "channels_with_scenario": sensitivity_channels,
            "wins_vs_primary": sensitivity_wins,
            "losses_vs_primary": sensitivity_losses,
            "ties_vs_primary": sensitivity_ties,
            "mean_smape_delta_overlay_minus_primary": (
                sum(sensitivity_smape_deltas) / len(sensitivity_smape_deltas)
                if sensitivity_smape_deltas else None),
            "overlay_metrics_mean": {
                key: sum(values) / len(values)
                for key, values in sorted(sensitivity_metrics_acc.items())
                if values and key in ("MAPE", "MAE", "RMSE", "SMAPE",
                                      "OW_sMAPE", "OW_RMSSE", "OW_MASE")
            },
            "warning": (
                "Retrospective diagnostic only. The governed primary path "
                "is the submitted forecast and the only headline score; "
                "overlay wins do not define a deployment selection policy."
            ),
        },
        "best_effort": args.best_effort,
        "mcp_profile": args.mcp_profile if args.condition == "gnomon-mcp" else None,
        "compile_context": args.compile_context if args.condition == "gnomon-mcp" else None,
        "context_receipts_dir": (args.context_receipts_dir
                                 if args.condition == "gnomon-mcp" else None),
        "compile_questions": (args.compile_questions
                              if args.condition == "gnomon-mcp" else None),
        "question_receipts_dir": (args.question_receipts_dir
                                  if args.condition == "gnomon-mcp" else None),
        "model_evidence_registry": (args.model_evidence_registry
                                    if args.condition == "gnomon-mcp" else None),
        **({"mcp_economics": {
            "cumulative_tokens": mcp_run_tokens,
            "mean_tokens_per_attempted_row": round(
                mcp_run_tokens / len(mcp_calls_seen), 2),
            "calls_median": sorted(mcp_calls_seen)[len(mcp_calls_seen) // 2],
            "calls_p95": sorted(mcp_calls_seen)[
                max(0, (95 * len(mcp_calls_seen) + 99) // 100 - 1)],
            "schema_bytes": sorted(mcp_schema_bytes),
            "rows_answered": mcp_rows_answered,
            "rows_attempted": len(mcp_calls_seen),
            "answer_yield": round(
                mcp_rows_answered / len(mcp_calls_seen), 4),
            **({"compiler_calls": compiler_calls,
                "compiler_receipts_reused": compiler_receipts_reused,
                "compiler_events_accepted": compiler_events,
                "compiler_hypotheses_accepted": compiler_hypotheses,
                "compiler_proposals_rejected": compiler_rejected,
                "context_channels_with_engine_receipt": context_channels_considered,
                "context_events_admitted_by_engine": context_events_admitted,
                "context_events_rejected_by_engine": context_events_rejected,
                "context_events_applied_to_forecasts": context_events_applied,
                "context_events_published_as_scenario_only":
                    context_events_scenario_only}
               if args.compile_context else {}),
            **({"question_compiler_calls": question_compiler_calls,
                "question_compiler_receipts_reused":
                    question_compiler_receipts_reused,
                "questions_accepted": question_compiler_accepted,
                "question_proposals_rejected": question_compiler_rejected,
                "temporal_answer_receipts": temporal_answer_receipts,
                "temporal_answers_returned": temporal_answers_returned,
                "answer_receipts_primary_forecast_unchanged":
                    temporal_primary_unchanged,
                "choice_reasoning_stages": {
                    "questions_requested": typed_questions_requested,
                    "questions_with_engine_answer":
                        typed_questions_with_engine_answer,
                    "compiler_to_engine_coverage": round(
                        typed_questions_with_engine_answer
                        / typed_questions_requested, 4)
                        if typed_questions_requested else None,
                    "officially_correct_with_engine_answer":
                        typed_engine_answers_officially_correct,
                    "engine_answers_comparable_to_official_options":
                        typed_engine_answers_officially_comparable,
                    "official_accuracy_conditional_on_engine_answer": round(
                        typed_engine_answers_officially_correct
                        / typed_engine_answers_officially_comparable, 4)
                        if typed_engine_answers_officially_comparable else None,
                    "engine_answers_comparable_to_agent_submission":
                        typed_answers_comparable_to_submission,
                    "agent_preserved_canonical_answer":
                        typed_answers_preserved_by_agent,
                    "agent_preservation_rate": round(
                        typed_answers_preserved_by_agent
                        / typed_answers_comparable_to_submission, 4)
                        if typed_answers_comparable_to_submission else None,
                    "note": ("Official accuracy measures the submitted task "
                             "answer; preservation only compares exact canonical "
                             "or explicitly projected display values."),
                }}
               if args.compile_questions else {}),
            **({
                "future_covariate_channels_considered": covariate_channels_considered,
                "future_covariate_channels_admitted": covariate_channels_admitted,
            } if forecast_rows_total else {}),
        }} if mcp_calls_seen else {}),
        "forecast_channel_support_mix": dict(sorted(support_mix.items())),
        "forecast_channels_abstained": channels_abstained,
        **({"forecast_channel_routes": dict(sorted(route_mix.items()))}
           if route_mix else {}),
        "note": (
            "forecast metrics computed by the dataset's official "
            "forecast_metrics_utils.py; choice accuracy is this adapter's "
            "LOCAL case-insensitive exact match against the official "
            "labels. The *_scored_only means average scored rows only — "
            "fully-abstained and errored rows are not in them (rows with "
            "PARTIAL channel abstentions are, scored on the channels "
            "present, and Uncertain/sentinel choice answers count as "
            "wrong). Rows the harness voided (rows_voided_by_harness — a "
            "breached cap, no submission) are out of the choice "
            "denominators too: an answer the harness never collected is "
            "not a wrong answer. So they are unmatched-subset numbers; "
            "compare arms through "
            "the per-channel path (benchmarks/temporalbench/"
            "score_per_channel.py) or benchmarks/report.py's matched "
            "join, not by subtracting summaries. success on T2/T4 "
            "records completion (the official module returned metrics), "
            "not accuracy — per-row SMAPE lives in each record's extra, and "
            "every record names what its success measured in success_basis "
            "(completion vs all_choices_correct); voided rows carry "
            "row_abstained, which report.py excludes from its success test. "
            "forecast_channel_support_mix and forecast_channels_abstained "
            "are the coverage every forecast figure rests on; quote them "
            "beside it. Any best_effort count in the mix is disclosed "
            "fallback rows (NO RELIABLE FORECAST), not supported "
            "forecasts; a 'model' count is gnomon-mcp channels the model "
            "wrote itself (see forecast_channel_routes)."
        ),
    }
    if client is not None:
        current_usage = client.usage_summary
        prior_usage = (prior_summary.get("llm_usage") or {}) if args.resume else {}
        cumulative = dict(current_usage)
        for key in ("requests", "transport_attempts", "prompt_tokens",
                    "completion_tokens", "cost_usd",
                    "truncation_escalations"):
            cumulative[key] = (prior_usage.get(key, 0)
                               + current_usage.get(key, 0))
        summary["llm_usage"] = cumulative
        summary["llm_usage_this_invocation"] = current_usage
        if prior_summary.get("merged_usage_sources"):
            summary["merged_usage_sources"] = prior_summary[
                "merged_usage_sources"]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "usage.checkpoint.json").unlink(missing_ok=True)
    # Provenance beside the results on direct CLI runs too, mirroring
    # run_all.py's field conventions (run_all overwrites this with its
    # own manifest when it is the caller).
    write_manifest(
        output_dir,
        benchmark="temporalbench",
        condition=args.condition,
        model=args.model,
        target="tiers=" + ",".join(tiers or TIERS)
               + (";datasets=" + ",".join(datasets) if datasets else ""),
        command=" ".join(sys.argv),
        limit=args.limit,
        # Which endpoint served the model: not part of `target` (it does
        # not change the task set), but it does change what the score is
        # a measurement of, so it belongs in provenance.
        base_url=client.base_url if client is not None else None,
        request_timeout=args.request_timeout if client is not None else None,
        max_retries=args.max_retries if client is not None else None,
        infrastructure_retries=args.infrastructure_retries,
        resume=args.resume or None,
        retry_voided=args.retry_voided or None,
        # Not part of `target`: best_effort changes the condition's
        # behaviour, not the task set, so it must not make report.py
        # refuse a control-vs-treatment join. It still has to be visible
        # in provenance, hence its own field (None keeps old manifests
        # byte-identical).
        best_effort=args.best_effort or None,
        named_tsfm=args.named_tsfm,
        mcp_profile=(args.mcp_profile
                     if args.condition == "gnomon-mcp" else None),
        compile_context=(args.compile_context
                         if args.condition == "gnomon-mcp" else None),
        context_receipts_dir=(args.context_receipts_dir
                              if args.condition == "gnomon-mcp" else None),
        compile_questions=(args.compile_questions
                           if args.condition == "gnomon-mcp" else None),
        question_receipts_dir=(args.question_receipts_dir
                               if args.condition == "gnomon-mcp" else None),
        model_evidence_registry=(args.model_evidence_registry
                                 if args.condition == "gnomon-mcp" else None),
    )
    print(json.dumps(summary, indent=2))
    # A fully failed run has produced diagnostics, not benchmark evidence.
    # Returning success here allowed stale compiler receipts (and similar
    # deterministic setup failures) to look like a completed zero-score run.
    return 1 if total and errored == total else 0


if __name__ == "__main__":
    raise SystemExit(main())
