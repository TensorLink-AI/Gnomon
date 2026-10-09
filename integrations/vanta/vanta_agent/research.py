"""Offline strategy research: champion vs challenger on a frozen dataset.

The loop is deliberately incremental:
- `snapshot` freezes market data (CSV + manifest with hashes). Every result names
  the dataset it came from.
- The dataset's decision hours split into a TUNE window (iterate freely) and a later
  HOLDOUT window (locked: looked at only to promote, a limited number of times).
- `compare` / `sweep` evaluate a challenger (usually one parameter changed from the
  champion) on the tune window. Both strategies trade the same hours, so the hourly
  net-return differences are paired; a moving-block bootstrap gives a one-sided lower
  bound on the improvement. A sweep of k candidates uses alpha/k.
- `promote` re-checks the accepted tune result, spends one holdout look, and only
  then replaces the champion strategy file.
Every evaluation is appended to trials.jsonl, so the number of looks behind any
result stays visible. Forecasts are cached, so re-evaluating policy changes is free.
"""
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import shutil

from . import strategy as strategy_mod
from .backtest import ForecastCache, aligned_length, forecast_panel, forecast_quality, simulate
from .market_data import MarketData, hourly, load_csv
from .signals import open_session

WARMUP_HOURS = 168  # first decision needs this much history (shared by all strategies)


# -- dataset -----------------------------------------------------------------------
def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def snapshot(config, *, hours, now=None, market=None):
    """Freeze `hours` of 5m candles per pair into the research dataset directory."""
    now = now or datetime.now(timezone.utc)
    market = market or MarketData(config)
    out = config.path(config.research.dataset_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for pair in config.universe.pairs:
        candles = market.candles(pair, hours=hours, now=now)
        rows = ["open_time,close"] + [f"{c.open_time.isoformat()},{c.close!r}" for c in candles]
        (out / f"{pair}.csv").write_text("\n".join(rows) + "\n")
        files[pair] = dict(sha256=_sha(out / f"{pair}.csv"), bars=len(candles),
                           first=candles[0].open_time.isoformat() if candles else None,
                           last=candles[-1].open_time.isoformat() if candles else None)
    manifest = dict(created_at=now.isoformat(), source=config.market_data.source, files=files)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return dict(dataset_id=dataset_id(config), **manifest)


def dataset_id(config):
    manifest = json.loads((config.path(config.research.dataset_dir) / "manifest.json").read_text())
    return sha256(json.dumps({p: f["sha256"] for p, f in sorted(manifest["files"].items())}).encode()).hexdigest()[:12]


def load_dataset(config):
    directory = config.path(config.research.dataset_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    series = {}
    for pair in config.universe.pairs:
        meta = manifest["files"].get(pair)
        if meta is None:
            raise ValueError(f"{pair} is not in the dataset snapshot; run `research snapshot`")
        if _sha(directory / f"{pair}.csv") != meta["sha256"]:
            raise ValueError(f"{pair}.csv changed since the snapshot; snapshot again instead of editing data")
        series[pair] = hourly(load_csv(directory / f"{pair}.csv"))
    return series


def windows(config, series):
    """Decision indices of the tune and holdout windows (last hour has no next close)."""
    length = aligned_length(series)
    decisions = list(range(WARMUP_HOURS, length - 1))
    if len(decisions) < 48:
        raise ValueError(f"Only {len(decisions)} decision hours after {WARMUP_HOURS}h warmup; snapshot more data")
    cut = int(len(decisions) * config.research.tune_fraction)
    return {"tune": decisions[:cut], "holdout": decisions[cut:]}


# -- statistics ---------------------------------------------------------------------
def block_bootstrap_lcb(diffs, *, block, samples, alpha, seed):
    """One-sided lower confidence bound of the mean of `diffs` (moving-block bootstrap)."""
    n = len(diffs)
    if n == 0:
        return 0.0
    block = max(1, min(block, n))
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        total, taken = 0.0, 0
        while taken < n:
            start = rng.randrange(0, n - block + 1)
            chunk = diffs[start:start + min(block, n - taken)]
            total += sum(chunk)
            taken += len(chunk)
        means.append(total / n)
    means.sort()
    return means[max(0, min(samples - 1, int(math.floor(alpha * samples))))]


# -- the research session ------------------------------------------------------------
class Lab:
    def __init__(self, config, *, max_remote_calls=0):
        self.config = config
        self.work = config.path(config.research.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.series = load_dataset(config)
        self.dataset = dataset_id(config)
        self.windows = windows(config, self.series)
        self.max_remote_calls = max_remote_calls
        self._panels = {}
        self._sessions = {}

    # forecasts depend only on [forecast]; risk changes reuse the panel
    def panel(self, config, window):
        key = (json.dumps(strategy_mod.policy_params(config)["forecast"], sort_keys=True, default=list), window)
        if key not in self._panels:
            providers = config.forecast.providers_config
            if providers not in self._sessions:
                session = open_session(config)
                self._sessions[providers] = ForecastCache(session, self.work / "forecast_cache.sqlite",
                                                          max_remote_calls=self.max_remote_calls)
            cache = self._sessions[providers]
            self._panels[key] = forecast_panel(cache, config, self.series, self.windows[window])
        return self._panels[key]

    @property
    def remote_calls(self):
        return sum(c.remote_calls for c in self._sessions.values())

    def close(self):
        for cache in self._sessions.values():
            cache.session.close()

    # trial journal
    @property
    def trials_path(self):
        return self.work / "trials.jsonl"

    def trials(self):
        return read_trials(self.trials_path)

    def _log(self, record):
        record = dict(time=datetime.now(timezone.utc).isoformat(), dataset=self.dataset, **record)
        with self.trials_path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return record

    def evaluate(self, config, window="tune"):
        panel = self.panel(config, window)
        summary, rets = simulate(panel, config)
        return dict(revision=config.revision, strategy=summary, forecast_quality=forecast_quality(panel),
                    vol_sized_long=simulate(panel, config, reference="vol_long")[0]), rets

    def compare(self, champion, challenger, *, window="tune", alpha=None, log_kind="compare"):
        r = self.config.research
        alpha = r.alpha if alpha is None else alpha
        base, base_rets = self.evaluate(champion, window)
        cand, cand_rets = self.evaluate(challenger, window)
        diffs = [b - a for a, b in zip(base_rets, cand_rets)]
        seed = int(sha256(f"{self.dataset}:{window}:{champion.revision}:{challenger.revision}".encode()).hexdigest()[:8], 16)
        lcb = block_bootstrap_lcb(diffs, block=r.block_hours, samples=r.bootstrap_samples, alpha=alpha, seed=seed)
        mean = sum(diffs) / len(diffs) if diffs else 0.0
        per_day = lambda x: round(x * 24 * 1e4, 3)  # bps of equity per day
        if challenger.revision == champion.revision:
            verdict, reason = "same", "identical policy parameters"
        elif cand["strategy"]["vanta_eliminated"]:
            verdict, reason = "reject", "challenger would have been eliminated by Vanta"
        elif (cand["strategy"]["max_intraday_drawdown"] >= challenger.risk.intraday_drawdown_halt or
              cand["strategy"]["max_eod_drawdown"] >= challenger.risk.eod_drawdown_halt):
            verdict, reason = "reject", "challenger crossed the agent's own drawdown halt lines"
        elif window == "holdout":
            ok = mean >= 0
            verdict, reason = ("pass" if ok else "fail"), "holdout mean improvement " + ("≥ 0" if ok else "< 0")
        elif lcb > 0:
            verdict, reason = "accept", f"lower bound of improvement > 0 at alpha={alpha:g}"
        else:
            verdict, reason = "reject", f"improvement not distinguishable from zero at alpha={alpha:g}"
        return self._log(dict(kind=log_kind, window=window, hours=len(diffs), alpha=alpha,
                              champion=champion.revision, challenger=challenger.revision,
                              change=strategy_mod.diff(champion, challenger),
                              mean_improvement_bps_per_day=per_day(mean), lcb_bps_per_day=per_day(lcb),
                              verdict=verdict, reason=reason, champion_result=base, challenger_result=cand))

    def sweep(self, champion, grid):
        """One parameter at a time from the champion; alpha/k across the k candidates."""
        candidates = []
        for key, values in grid.items():
            for value in values:
                child = strategy_mod.with_overrides(champion, {key: value})
                if child.revision != champion.revision:
                    candidates.append(child)
        alpha = self.config.research.alpha / max(1, len(candidates))
        results = []
        for child in candidates:
            path = strategy_mod.dump(child, self.work / "candidates" / f"{child.revision.replace('@', '-')}.toml",
                                     description=f"sweep from {champion.revision}")
            record = self.compare(champion, child, alpha=alpha, log_kind="sweep")
            results.append(dict(candidate=str(path), **{k: record[k] for k in (
                "challenger", "change", "mean_improvement_bps_per_day", "lcb_bps_per_day", "verdict")}))
        results.sort(key=lambda row: row["lcb_bps_per_day"], reverse=True)
        return dict(champion=champion.revision, dataset=self.dataset, candidates=len(candidates),
                    alpha_per_candidate=alpha, results=results, remote_calls=self.remote_calls)

    def holdout_looks(self):
        return sum(1 for t in self.trials() if t.get("kind") == "holdout" and t.get("dataset") == self.dataset)

    def promote(self, champion, challenger, champion_path):
        """Spend a holdout look on an accepted challenger; on a pass, it becomes the champion."""
        if challenger.revision == champion.revision:
            raise ValueError(f"{challenger.revision} is already the champion")
        accepted = [t for t in self.trials() if t.get("dataset") == self.dataset and t.get("window") == "tune"
                    and t.get("champion") == champion.revision and t.get("challenger") == challenger.revision]
        if not accepted or accepted[-1]["verdict"] != "accept":
            raise ValueError(f"No accepted tune comparison of {challenger.revision} against the current champion "
                             f"{champion.revision} on dataset {self.dataset}; run `research compare` first")
        looks = self.holdout_looks()
        if looks >= self.config.research.max_holdout_looks:
            raise ValueError(f"The holdout of dataset {self.dataset} has been looked at {looks} times; it no longer "
                             "tests anything. Snapshot newer data (`research snapshot`) and re-run the comparison.")
        record = self.compare(champion, challenger, window="holdout", log_kind="holdout")
        if record["verdict"] != "pass":
            return dict(promoted=False, **_short(record))
        champion_path = Path(champion_path)
        archive = self.work / "champions"
        archive.mkdir(exist_ok=True)
        if champion_path.exists():
            shutil.copy(champion_path, archive / f"{champion.revision.replace('@', '-')}.toml")
        promoted = replace(challenger, strategy_parent=champion.revision)
        strategy_mod.dump(promoted, champion_path, description=f"promoted over {champion.revision} on dataset "
                          f"{self.dataset}: {record['change']}")
        self._log(dict(kind="promote", champion=champion.revision, challenger=challenger.revision,
                       change=record["change"], verdict="promoted"))
        return dict(promoted=True, champion_file=str(champion_path), **_short(record))


def _short(record):
    return {k: record[k] for k in ("window", "champion", "challenger", "change", "mean_improvement_bps_per_day",
                                   "lcb_bps_per_day", "verdict", "reason")}


def read_trials(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_trials(trials, dataset=None):
    rows = [t for t in trials if dataset is None or t.get("dataset") == dataset]
    by_kind = {}
    for t in rows:
        by_kind[t["kind"]] = by_kind.get(t["kind"], 0) + 1
    return dict(dataset=dataset, evaluations=by_kind,
                promotions=[{k: t.get(k) for k in ("time", "champion", "challenger", "change")}
                            for t in rows if t["kind"] == "promote"],
                last=[_short(t) for t in rows if t["kind"] in ("compare", "sweep", "holdout")][-10:],
                note="Each tune comparison is a look at the same data: many looks inflate the chance that "
                     "an accepted change is noise. The holdout and paper trading are the checks on that.")
