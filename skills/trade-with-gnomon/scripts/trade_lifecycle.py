"""Offline synthetic lifecycle; explicit fixture clock, no orders or network calls.

Run: python trade_lifecycle.py /path/to/a/new/output-directory
The directory must not exist, to keep each fixture run isolated.

This file is a demo harness. To build a strategy, reuse trade_decisions.py and
replace everything here: synthetic data, toy providers, the q10 policy, regime rule,
cost model and fixture clock. Never copy FixedClock or clock reassignment into a
real ledger workflow.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from math import exp, expm1, log, log1p
from pathlib import Path
from statistics import fmean, quantiles
import sys

from gnomon import AdapterCapabilities, ForecastResult, GnomonSession, TemporalLedger
from gnomon.ids import FixedClock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trade_decisions import client_order_id, record_trade_decision  # noqa: E402


def empirical_returns(request):
    # Illustrative historical marginal quantiles; not calibrated predictions.
    deciles = quantiles(request.history, n=10, method="inclusive")
    return forecast_result(request, (deciles[0], deciles[4], deciles[8]))


def forecast_result(request, values):
    return ForecastResult(
        point=(values[1],), quantiles=({q: v for q, v in zip((.1, .5, .9), values)},),
        timestamps=request.future_timestamps, series_id=request.series_id, unit=request.unit,
    )


def choose_intent(q10, cost_hurdle):
    # Toy policy: long only when the forecast's 0.1 log-return quantile clears costs.
    return "LONG" if q10 > log1p(cost_hurdle) else "NO_NEW_POSITION"


def classify_regime(history):
    return "positive-trailing-24h" if fmean(history[-24:]) > 0 else "nonpositive-trailing-24h"


def assess_outcome(action, realised_log_return, cost_hurdle):
    log_hurdle = log1p(cost_hurdle)
    relation = "exceeded" if realised_log_return > log_hurdle else (
        "missed" if realised_log_return < log_hurdle else "matched")
    gross = expm1(realised_log_return)
    # Toy cost model: round-trip costs are a fixed fraction of entry notional.
    net_if_long = gross - cost_hurdle
    return dict(hypothetical=True, cost_model="fixed_fraction_of_entry_notional",
                cost_hurdle=cost_hurdle, log_cost_hurdle=log_hurdle,
                observed_log_return=realised_log_return, hurdle_relation=relation,
                gross_simple_return=gross, net_simple_return_if_long=net_if_long,
                policy_net_simple_return=net_if_long if action == "LONG" else 0.0)


def main(output, *, observed_log_return=.0004):
    output.mkdir(parents=True, exist_ok=False)
    origin = datetime(2026, 1, 20, 12, tzinfo=timezone.utc)
    target = origin + timedelta(hours=1)
    ledger = TemporalLedger(output / "ledger.db", clock=FixedClock(origin))
    prices = [100.0]
    for i in range(60):
        prices.append(prices[-1] * exp(.0014 + .0001 * (i % 5)))
    history = [log(b / a) for a, b in zip(prices, prices[1:])]
    regime = classify_regime(history)  # Frozen rule: trailing 24 completed hourly returns.
    request = dict(
        history=history, horizon=1, quantiles=[.1, .5, .9], frequency="h",
        series_id="synthetic/spot/log-return/1h", unit="log_return",
        timestamps=[(origin - timedelta(hours=59-i)).isoformat() for i in range(60)],
        future_timestamps=[target.isoformat()], cutoff=origin.isoformat(),
    )
    # Policy fixture: flat, long-only, one-hour holding horizon; no pyramiding.
    # All-in break-even gross simple-return hurdle; replace with a venue cost model.
    cost_hurdle = .001
    with GnomonSession.from_config(ledger=ledger) as session:
        cap = AdapterCapabilities(quantiles=True, min_history=2, max_horizon=1)
        session.engine.register("empirical", empirical_returns, capabilities=cap, revision="empirical-v1")
        session.engine.register("zero-return", lambda r: forecast_result(r, (0., 0., 0.)),
                                capabilities=cap, revision="zero-return-v1")
        runs = [session.forecast(p, request) for p in ("empirical", "zero-return")]
        forecast = runs[0]
        q10 = forecast["result"]["quantiles"][0][.1]
        action = choose_intent(q10, cost_hurdle)
        intent = record_trade_decision(
            ledger, output / "intent.json", forecast=forecast, action=action,
            account="paper-demo", leg="entry-0", event_id=f"paper-demo:{origin.isoformat()}",
            policy_revision="q10-cost-v1", mode="backtest",
            rationale=f"policy=q10-cost-v1; intent={action}; q10={q10}; hurdle={log1p(cost_hurdle)}",
            assumptions=["Synthetic fixture; flat account; returns and cost hurdle share a one-hour horizon."],
            invalidation_conditions=["Data stale or account not flat: cancel entry intent.",
                                     "At horizon, exit under the separately configured execution policy."],
            context=[dict(key="regime", value=regime, valid_from=origin.isoformat(),
                          valid_to=target.isoformat(), source_available_at=origin.isoformat(),
                          source_ref="fixture:generated-history-v1/trailing-24h-mean-sign-v1")],
            evidence_refs=[{"kind": "execution", "id": r["execution_id"]} for r in runs],
            require_system_clock=False,  # Demo only: the fixture clock simulates time passing.
        )
        decision_id = intent["decision_id"]
        early = dict(source_as_of=origin.isoformat(), recorded_as_of=origin.isoformat())
        assert not ledger.review_decision(decision_id=decision_id, **early)["review_ready"]

        arrived = target + timedelta(minutes=5)
        ledger.clock = FixedClock(arrived)  # Test-only time advance, not production backdating.
        observed_price = prices[-1] * exp(observed_log_return)
        actual = dict(series_id=request["series_id"], unit=request["unit"],
                      valid_time=target.isoformat(), value=log(observed_price / prices[-1]),
                      source_available_at=arrived.isoformat(), source_ref="fixture:close-v1")
        ledger.append_actual(**actual)
        cutoffs = dict(source_as_of=arrived.isoformat(), recorded_as_of=arrived.isoformat())
        review = ledger.review_decision(decision_id=decision_id, **cutoffs)
        assert review["review_ready"]
        scope = dict(series_id=request["series_id"], unit=request["unit"], horizon=1,
                     providers={r["provider"]: r["revision"] for r in runs},
                     start=origin.isoformat(), end=origin.isoformat(), metric="mae")
        comparison = ledger.compare_context(context_filters={"regime": regime},
                                            **scope, **cutoffs)
        assert comparison["matched_origins"] == 1
        ranking = comparison["evidence_summary"]["lifetime"]["ranking"]
        outcome = assess_outcome(action, actual["value"], cost_hurdle)
        lesson = ledger.record_lesson(
            decision_id=decision_id,
            lesson=f"The toy policy intent was {action}; the observed return {outcome['hurdle_relation']} "
                   f"the cost hurdle. Hypothetical policy net simple return: {outcome['policy_net_simple_return']:.8f}. "
                   "One synthetic origin establishes neither calibration nor strategy performance.",
            **cutoffs,
        )
        exported = ledger.export_lesson(lesson_id=lesson["lesson_id"], recorded_as_of=arrived.isoformat())

        revised_at = arrived + timedelta(hours=1)
        ledger.clock = FixedClock(revised_at)
        ledger.append_actual(**{**actual, "value": .0007, "source_available_at": revised_at.isoformat(),
                                "source_ref": "fixture:close-v2"})
        later = dict(source_as_of=revised_at.isoformat(), recorded_as_of=revised_at.isoformat())
        revised = ledger.review_decision(decision_id=decision_id, **later)
        assert ledger.review_decision(decision_id=decision_id, **cutoffs) == review
        assert revised["actual_ids"] != review["actual_ids"]
        assert ledger.export_lesson(lesson_id=lesson["lesson_id"], recorded_as_of=revised_at.isoformat()) == exported
        saved_intent = json.loads((output / "intent.json").read_text())
        assert client_order_id("paper-demo", saved_intent["decision_id"], "entry-0") == intent["client_order_id"]
        result = dict(fixture=True, intent=intent, review=review, revised_review=revised,
                      comparison=comparison, outcome=outcome, lesson=exported)
        (output / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(output=str(output), action=action, decision_id=decision_id,
                              matched_origins=comparison["matched_origins"],
                              metric=comparison["metric"], ranking=ranking,
                              outcome=outcome, orders_submitted=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for synthetic artifacts")
    main(parser.parse_args().output)
