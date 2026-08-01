"""Run TemporalBench (arXiv:2602.13272) conditions and score them with
the official metric module shipped in the dataset.

Conditions
----------
``control``     the row's official prompt sent verbatim to an OpenRouter
                model; its JSON answer is scored as-is.
``aion-pure``   T2/T4 only, no LLM: Aion forecasts every target channel;
                multiple-choice answers are 'Uncertain' (an honest
                abstention the option sets allow).
``aion-agent``  Aion computes the evidence (per-channel forecasts,
                season, anomalies, stats); the LLM sees the official
                prompt plus that evidence and answers only the choice
                questions. Forecast arrays in the final answer are the
                Aion arrays — the model cannot edit them.

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
        --data-dir ~/temporalbench --condition aion-agent \
        --model openai/gpt-4o --tiers T2,T4 --limit 50 \
        --output-dir results/tb-aion

    aion eval compare --baseline results/tb-control/aionbench.jsonl \
                      --treatment results/tb-aion/aionbench.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402
from benchmarks.common.records import RecordWriter, RunRecord  # noqa: E402
from benchmarks.temporalbench import aion_runner, scoring  # noqa: E402
from benchmarks.temporalbench.tasks import (  # noqa: E402
    TIERS,
    download,
    extract_json_object,
    iter_rows,
    load_official_metrics,
)

AGENT_PREAMBLE = """\
Deterministic tool evidence computed from the task's own data by the
Aion engine (backtested forecasts, graded anomaly detection, season
detection). Base every numeric judgement on this evidence; where a
forecast is required in the output, reproduce these arrays exactly.

<aion_evidence>
{evidence}
</aion_evidence>

"""


def answer_row(row: dict[str, Any], condition: str,
               client: OpenRouterClient | None) -> dict[str, Any]:
    """Produce the row's answer object under the given condition."""
    if condition == "control":
        completion = client.completions(
            [{"role": "user", "content": row["prompt"]}], n=1
        )[0]
        return {"answer": extract_json_object(completion), "abstained": []}

    analysis = aion_runner.analyse_row(row)
    tier = row.get("tier")
    if condition == "aion-pure":
        if tier not in ("T2", "T4"):
            raise ValueError("aion-pure covers tiers T2 and T4 only")
        forecast, abstained = aion_runner.forecast_payload(analysis)
        return {
            "answer": {"forecast": forecast,
                       "mcq": aion_runner.uncertain_mcq(row)},
            "abstained": abstained, "analysis": analysis,
        }

    # aion-agent: evidence digest + official prompt; LLM answers choices.
    digest = {k: v for k, v in analysis.items() if k != "channels"}
    digest["forecasts"] = {
        key: (outcome if outcome.get("abstained") else
              {"support": outcome["support"],
               "selected_model": outcome["selected_model"],
               "values": outcome["values"]})
        for key, outcome in analysis.get("channels", {}).items()
    }
    prompt = AGENT_PREAMBLE.format(
        evidence=json.dumps(digest)[:40_000]
    ) + row["prompt"]
    completion = client.completions(
        [{"role": "user", "content": prompt}], n=1
    )[0]
    answer = extract_json_object(completion)
    abstained: list[str] = []
    if tier in ("T2", "T4"):
        forecast, abstained = aion_runner.forecast_payload(analysis)
        answer["forecast"] = forecast  # Aion owns the numbers.
    return {"answer": answer, "abstained": abstained, "analysis": analysis}


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
                        choices=["control", "aion-pure", "aion-agent"])
    parser.add_argument("--model", default=None, help="OpenRouter model id")
    parser.add_argument("--tiers", default="T1,T2,T3,T4")
    parser.add_argument("--datasets", default=None,
                        help="Comma list of source datasets to keep")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.download:
        download(data_dir)
        print(f"Dataset ready under {data_dir}")
        if not args.condition:
            return 0
    if not args.condition or not args.output_dir:
        parser.error("--condition and --output-dir are required to run")
    if args.condition != "aion-pure" and not args.model:
        parser.error("--model is required for this condition")

    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip() in TIERS)
    if args.condition == "aion-pure":
        tiers = tuple(t for t in tiers if t in ("T2", "T4")) or ("T2", "T4")
    datasets = (tuple(d.strip() for d in args.datasets.split(","))
                if args.datasets else None)
    official_metrics = load_official_metrics(data_dir)
    client = (OpenRouterClient(args.model, temperature=args.temperature,
                               max_tokens=8000)
              if args.model else None)

    output_dir = Path(args.output_dir)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    records = RecordWriter(output_dir / "aionbench.jsonl")

    choice_by_tier: dict[str, list[int]] = {}
    forecast_metrics_acc: dict[str, list[float]] = {}
    abstained_rows = errored = 0
    total = 0
    for row in iter_rows(data_dir, tiers=tiers or TIERS,
                         datasets=datasets, limit=args.limit):
        total += 1
        row_id = row.get("id", f"row{total}")
        started = time.time()
        try:
            outcome = answer_row(row, args.condition, client)
            verdict = score_row(row, outcome["answer"], official_metrics)
        except Exception as error:
            errored += 1
            records.write(RunRecord(task_id=row_id, success=False,
                                    extra={"error": str(error)[:400]}))
            continue
        elapsed = time.time() - started

        choice = verdict.get("choice") or {}
        tier = verdict["tier"]
        if choice.get("total"):
            choice_by_tier.setdefault(tier, []).extend(
                [1] * choice["correct"] + [0] * (choice["total"] - choice["correct"])
            )
        metrics = verdict.get("forecast_metrics")
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    forecast_metrics_acc.setdefault(key, []).append(float(value))
        if outcome.get("abstained"):
            abstained_rows += 1

        success = bool(metrics) if tier in ("T2", "T4") else (
            choice.get("total", 0) > 0 and choice["correct"] == choice["total"]
        )
        records.write(RunRecord(
            task_id=row_id, success=success,
            appropriate_abstention=bool(outcome.get("abstained")),
            tool_calls=0 if args.condition == "control" else 1,
            latency_seconds=round(elapsed, 3),
            extra={"tier": tier,
                   "choice_correct": choice.get("correct"),
                   "choice_total": choice.get("total"),
                   "smape": (metrics or {}).get("SMAPE")
                   or (metrics or {}).get("OW_sMAPE")},
        ))
        (details_dir / f"{row_id}.json").write_text(
            json.dumps({"verdict": verdict,
                        "abstained": outcome.get("abstained"),
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
        "choice_accuracy_by_tier": {
            tier: sum(flags) / len(flags)
            for tier, flags in sorted(choice_by_tier.items())
        },
        "forecast_metrics_mean": {
            key: sum(values) / len(values)
            for key, values in sorted(forecast_metrics_acc.items())
            if values and key in ("MAPE", "MAE", "RMSE", "SMAPE",
                                  "OW_sMAPE", "OW_RMSSE", "OW_MASE")
        },
        "note": (
            "forecast metrics computed by the dataset's official "
            "forecast_metrics_utils.py; choice accuracy is exact-match "
            "against the official labels"
        ),
    }
    if client is not None:
        summary["llm_usage"] = client.usage_summary
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
