"""Gnomon-guided next steps: rank forecasts on REALISED outcomes, propose one change.

The paper (or live) ledger holds every forecast the agent made: the models it trades
on, their baselines and any shadow models, each with its forecast origin. Once the
actuals arrive, Gnomon's `compare` scores all executions at one origin against the
same actuals. Aggregated over origins, that is prospective evidence: none of it
could have been tuned on.

`guide` turns it into at most one suggestion per forecast role (volatility,
direction), each written as a candidate strategy file one parameter away from the
current strategy. A better forecast is not yet a better strategy, so each
suggestion goes through `research compare` (P&L after costs, offline) and paper
trading before promotion.
"""
from datetime import datetime, timezone
from hashlib import sha256

from gnomon import TemporalLedger

from . import strategy as strategy_mod
from .research import block_bootstrap_lcb


def _executions(ledger, series_id, horizon, now):
    items, cursor = [], None
    while True:
        page = ledger.search(series_id=series_id, horizon=horizon, limit=100, cursor=cursor,
                             source_as_of=now, recorded_as_of=now)
        items += page["items"]
        cursor = page["next_cursor"]
        if not cursor:
            return items


def score_role(ledger, series_ids, horizon, providers, now, *, in_bps_of_price=False):
    """Matched per-origin MAE for each provider; only origins where all are scorable.

    Price errors are converted to bps of the origin's last price, so pairs at very
    different price levels (BTC vs SOL) weigh equally when pooled.
    """
    per_origin = []
    for series_id in series_ids:
        by_origin = {}
        for item in _executions(ledger, series_id, horizon, now):
            if item["provider"] in providers and item.get("status") in ("ready", "scored"):
                by_origin.setdefault(item["origin"], {}).setdefault(item["provider"], item["execution_id"])
        for origin, runs in sorted(by_origin.items()):
            if len(runs) < 2:
                continue
            result = ledger.compare(list(runs.values()), source_as_of=now, recorded_as_of=now)
            scale = 1.0
            if in_bps_of_price:
                scale = 1e4 / ledger.execution(next(iter(runs.values())))["request"]["history"][-1]
            per_origin.append((series_id, origin, {m["provider"]: m["mae"] * scale for m in result["models"]}))
    return per_origin


def _versus(per_origin, current, other, research):
    """Paired evidence that `other` has lower MAE than `current` (positive = other better)."""
    diffs = [s[current] - s[other] for _, _, s in per_origin if current in s and other in s]
    if not diffs:
        return dict(origins=0)
    seed = int(sha256(f"{current}:{other}:{len(diffs)}".encode()).hexdigest()[:8], 16)
    lcb = block_bootstrap_lcb(diffs, block=research.block_hours, samples=research.bootstrap_samples,
                              alpha=research.alpha, seed=seed)
    return dict(origins=len(diffs), mean_mae_reduction=round(sum(diffs) / len(diffs), 4),
                lcb_mae_reduction=round(lcb, 4), win_rate=round(sum(d > 0 for d in diffs) / len(diffs), 3))


def guide(config, *, ledger_path=None, min_origins=48, now=None):
    f, research = config.forecast, config.research
    now = (now or datetime.now(timezone.utc)).isoformat()
    path = ledger_path or config.path(config.agent.state_dir) / f"{config.agent.mode}.db"
    ledger = TemporalLedger(path, create=False)
    pairs = config.universe.pairs
    roles = {
        "vol": dict(key="forecast.vol_model", current=f.vol_model, baseline=f.vol_baseline,
                    others=list(f.shadow_vol_models), horizon=1, unit="bps of realised vol",
                    series=[f"vanta/{p}/rv-1h" for p in pairs]),
        "direction": dict(key="forecast.direction_provider", current=f.direction_provider,
                          baseline=f.direction_baseline, others=list(f.shadow_direction_providers),
                          horizon=f.horizon_hours, unit="bps of price",
                          series=[f"vanta/{p}/close-1h" for p in pairs]),
    }
    report = dict(ledger=str(path), strategy=config.revision, as_of=now, min_origins=min_origins, roles={},
                  suggestions=[])
    for name, role in roles.items():
        providers = {role["current"], role["baseline"], *role["others"]}
        per_origin = score_role(ledger, role["series"], role["horizon"], providers, now,
                                in_bps_of_price=name == "direction")
        rows = {p: _versus(per_origin, role["current"], p, research) for p in sorted(providers - {role["current"]})}
        report["roles"][name] = dict(current=role["current"], metric=f"MAE, {role['unit']}",
                                     scored_origins=len(per_origin), versus_current=rows)
        better = [(p, r) for p, r in rows.items() if r.get("origins", 0) >= min_origins and r["lcb_mae_reduction"] > 0]
        if not better:
            enough = any(r.get("origins", 0) >= min_origins for r in rows.values())
            report["roles"][name]["finding"] = (
                f"{role['current']} is not beaten with confidence; keep it" if enough else
                f"fewer than {min_origins} matched origins; keep collecting (add shadow models to compare more)")
            continue
        provider, evidence = max(better, key=lambda pr: pr[1]["lcb_mae_reduction"])
        candidate = strategy_mod.with_overrides(config, {role["key"]: provider})
        path_out = strategy_mod.dump(candidate, config.path(research.work_dir) / "candidates" /
                                     f"{candidate.revision.replace('@', '-')}.toml",
                                     description=f"guide: {provider} beat {role['current']} on {evidence['origins']} "
                                                 f"realised origins in {path.name}")
        report["roles"][name]["finding"] = f"{provider} forecasts better than {role['current']}"
        report["suggestions"].append(dict(
            change={role["key"]: [role["current"], provider]}, evidence=evidence, candidate=str(path_out),
            next=f"python -m vanta_agent research compare --config <agent.toml> --candidate {path_out}",
            caution="Better forecasts are not yet better P&L: validate after costs offline, then paper-trade."))
    if not report["suggestions"]:
        report["next"] = ("No forecast change is supported by realised outcomes yet. Policy parameters "
                          "(sizing, hurdle, horizon) are tuned offline: `research sweep`.")
    return report
