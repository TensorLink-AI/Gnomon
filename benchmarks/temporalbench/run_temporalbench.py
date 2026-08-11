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
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
detection). Base every numeric judgement on this evidence; where a
forecast is required in the output, reproduce these arrays exactly.

<gnomon_evidence>
{evidence}
</gnomon_evidence>

"""

EVIDENCE_BUDGET = 40_000


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
               best_effort: bool = False) -> dict[str, Any]:
    """Produce the row's answer object under the given condition.

    ``channel_support`` in the result maps each forecast channel to its
    Gnomon support label; with ``best_effort`` enabled it is how a
    disclosed-fallback channel stays distinguishable from a supported
    one all the way into the details records and the summary.
    """
    if condition == "control":
        completion = client.completions(
            [{"role": "user", "content": row["prompt"]}], n=1
        )[0]
        return {"answer": extract_json_object(completion), "abstained": [],
                "channel_support": {}}

    if condition == "gnomon-mcp":
        from benchmarks.temporalbench.mcp_agent import mcq_row, run_row

        # best_effort is deliberately not passed: on this arm the model
        # itself decides whether to request the engine's labeled
        # fallback via the gnomon_forecast tool — the realistic path.
        # T1/T3 carry no forecast channels, so they take the same
        # session with the tier's own answer shape.
        if row.get("tier") in ("T2", "T4"):
            return run_row(row, client)
        return mcq_row(row, client)

    analysis = gnomon_runner.analyse_row(row, best_effort=best_effort)
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
    parser.add_argument("--tiers", default="T1,T2,T3,T4")
    parser.add_argument("--datasets", default=None,
                        help="Comma list of source datasets to keep")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None)
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
    client = (OpenRouterClient(args.model, temperature=args.temperature,
                               max_tokens=8000, base_url=args.base_url)
              if args.model else None)

    output_dir = Path(args.output_dir)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "gnomonbench.jsonl"
    # RecordWriter appends; a rerun into the same output dir must replace
    # the previous run's rows (as summary.json is), not accumulate them.
    records_path.unlink(missing_ok=True)
    records = RecordWriter(records_path)

    choice_by_tier: dict[str, list[int]] = {}
    choice_rows_by_tier: dict[str, int] = {}
    forecast_metrics_acc: dict[str, list[float]] = {}
    abstained_rows = errored = forecast_rows_scored = rows_voided = 0
    # Coverage beside every figure: how many channel forecasts each
    # number rests on, by support label, plus the abstained channels.
    support_mix: dict[str, int] = {}
    # gnomon-mcp: per-channel routes (gnomon / informed-direct / direct
    # / abstain) so the exits stay separable in analysis.
    route_mix: dict[str, int] = {}
    channels_abstained = 0
    total = 0
    # Denominator for the scored-only forecast means: every T2/T4 row the
    # run saw, so the coverage behind those means is a ratio in the
    # summary, not an inference from three separate counters.
    forecast_rows_total = 0
    for row in iter_rows(data_dir, tiers=tiers or TIERS,
                         datasets=datasets, limit=args.limit):
        total += 1
        if row.get("tier") in ("T2", "T4"):
            forecast_rows_total += 1
        row_id = row.get("id", f"row{total}")
        started = time.time()
        # Answering and scoring fail separately: a scoring exception must
        # not swallow an answer that abstained — the abstention flag has
        # to survive into the record either way.
        try:
            outcome = answer_row(row, args.condition, client,
                                 best_effort=args.best_effort)
        except Exception as error:
            errored += 1
            records.write(RunRecord(task_id=row_id, success=False,
                                    extra={"error": str(error)[:400],
                                           "error_stage": "answer"}))
            continue
        try:
            verdict = score_row(row, outcome["answer"], official_metrics)
        except Exception as error:
            errored += 1
            records.write(RunRecord(
                task_id=row_id, success=False,
                appropriate_abstention=bool(outcome.get("abstained")),
                extra={"error": str(error)[:400], "error_stage": "score",
                       "abstained": outcome.get("abstained")},
            ))
            continue
        elapsed = time.time() - started

        choice = verdict.get("choice") or {}
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
            choice_rows_by_tier[tier] = choice_rows_by_tier.get(tier, 0) + 1
            choice_by_tier.setdefault(tier, []).extend(
                [1] * choice["correct"] + [0] * (choice["total"] - choice["correct"])
            )
        metrics = verdict.get("forecast_metrics")
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
                      if channel_route else {})},
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
                        "mcp": outcome.get("mcp"),
                        "answer": outcome["answer"]}, indent=2,
                       default=str) + "\n",
            encoding="utf-8",
        )
        if total % 20 == 0:
            print(f"...{total} rows")

    summary = {
        "benchmark": "temporalbench",
        "condition": args.condition,
        "model": args.model,
        "tiers": list(tiers or TIERS),
        "datasets": list(datasets) if datasets else "all",
        "rows": total,
        "rows_errored": errored,
        "rows_with_abstentions": abstained_rows,
        # Rows the harness ended without an answer (a breached cap, a run
        # that never submitted). They are in `rows`, out of every
        # accuracy denominator, and quoted here so the coverage behind
        # the tier means is visible rather than inferred.
        "rows_voided_by_harness": rows_voided,
        "choice_accuracy_by_tier_scored_only": {
            tier: sum(flags) / len(flags)
            for tier, flags in sorted(choice_by_tier.items())
        },
        "choice_rows_scored_by_tier": dict(sorted(choice_rows_by_tier.items())),
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
        "best_effort": args.best_effort,
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
        summary["llm_usage"] = client.usage_summary
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
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
        # Not part of `target`: best_effort changes the condition's
        # behaviour, not the task set, so it must not make report.py
        # refuse a control-vs-treatment join. It still has to be visible
        # in provenance, hence its own field (None keeps old manifests
        # byte-identical).
        best_effort=args.best_effort or None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
