"""Generate preregistered cases for run_workflow; never run an agent."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace

from .reference import action, inverse_cdf, threshold_probability

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {"pedestrian_counts_daily": "D", "retail_sales_monthly": "MS",
           "sensor_temps_5min": "5min", "wiki_traffic_daily_log": "D"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def timestamps(frequency):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    if frequency == "MS":
        return [datetime(2025 + i // 12, i % 12 + 1, 1, tzinfo=timezone.utc) for i in range(26)]
    delta = timedelta(minutes=5) if frequency == "5min" else timedelta(days=1)
    return [start + i * delta for i in range(26)]


def csv_text(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def windows():
    for source, frequency in SOURCES.items():
        path = ROOT / "benchmarks/workflow/data" / (source + ".csv")
        values = [float(row["value"]) for row in csv.DictReader(io.StringIO(path.read_text()))]
        for j in range(10):
            offset = j * (len(values) - 24) // 9
            seed = hashlib.sha256(f"{source}{j}business-v1".encode()).digest()
            multiplier, addition = .75 + seed[0] / 255, seed[1] / 10
            transformed = [multiplier * x + addition for x in values[offset:offset + 24]]
            scale = max(statistics.fmean(abs(b-a) for a, b in zip(transformed[:15], transformed[1:16])), 1e-6)
            yield {"source": source, "frequency": frequency, "window": j,
                   "cluster": f"{source}-{j}", "offset": offset,
                   "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "multiplier": multiplier, "addition": addition,
                   "values": transformed, "scale": scale}


def portable_snapshot(rows, frequency, cutoff, text):
    """Publish identical vintage data in a convenient additional format to all arms."""
    from gnomon.contracts import DataSchema
    from gnomon.data import Observation
    from gnomon.datasets import LoadedDataset
    from gnomon.snapshot_files import save_snapshot
    from gnomon.temporal_store import Snapshot, TemporalObservation
    source = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    vintages = [TemporalObservation("series", "value", datetime.fromisoformat(t),
                 datetime.fromisoformat(k), v, revision=r, source_ref=source)
                for t, k, v, r in rows]
    snapshot = Snapshot(vintages, cutoff, source_ref=source)
    latest = [Observation(row.valid_time, row.value, "series") for row in snapshot.series("series", "value")]
    loaded = LoadedDataset(source, ["timestamp", "known_time", "value", "revision"],
                           {"series": latest}, frequency, "UTC",
                           DataSchema("timestamp", "value", None, frequency, "UTC"), snapshot, "value")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "history.gnomon"
        save_snapshot(SimpleNamespace(loaded=loaded, unit="scenario_units", repairs=()), path)
        return path.read_text()


def case(identifier, evaluation, w, question, available, numbers=(), choices=(), facts=(), oracle=None, tags=()):
    return {"schema_version": 2, "id": identifier, "kind": "synthetic",
            "domain": w["source"], "question": question,
            "available_at_cutoff": {"hide_case_id_from_model": True, "retain_tool_results": True, "files_by_reference": True, **available},
            "answer_schema": {"numbers": list(numbers), "choices": list(choices), "facts": list(facts)},
            "oracle": oracle or {},
            "tags": ["business-utility-v1", evaluation, "cluster:" + w["cluster"], *tags]}


def eval1(w):
    y, stamps, scale = w["values"], timestamps(w["frequency"]), w["scale"]
    for trap in (False, True):
        rows = []
        for i in range(20):
            known = i + (1 if trap and w["window"] % 2 == 0 and i in range(15, 19) else 0)
            if known <= 19:
                rows.append((stamps[i].isoformat(), stamps[known].isoformat(), y[i], 0))
            if trap and w["window"] % 2 and i in range(15, 18):
                rows.append((stamps[i].isoformat(), stamps[i+2].isoformat(), (y[i]+y[i+1])/2, 1))
        def latest(origin, causal):
            eligible = [r for r in rows if r[0] <= stamps[origin].isoformat()
                        and (not causal or r[1] <= stamps[origin].isoformat())]
            return max(eligible, key=lambda r: (r[0], r[1], r[3]))[2]
        replay = [latest(i-1, True) for i in range(16, 20)]
        forbidden = [latest(i-1, False) for i in range(16, 20)]
        final = latest(19, True)
        mae = statistics.fmean(abs(p-y[i]) for p, i in zip(replay, range(16, 20))) / scale
        numbers = {**{f"replay_{i+1}": p for i, p in enumerate(replay)},
                   **{f"h{i+1}": final for i in range(4)}, "reported_nmae": mae}
        text = csv_text(["timestamp", "known_time", "value", "revision"], rows)
        public = {"files": {"vintages.csv": text, "history.gnomon": portable_snapshot(rows, w["frequency"], stamps[19], text)},
                  "as_of": stamps[19].isoformat(), "scale": scale, "frequency": w["frequency"],
                  "replay_origins": [stamps[i-1].isoformat() for i in range(16, 20)],
                  "replay_actuals": y[16:20], "unit": "scenario_units"}
        q = ("Audit this export as of the supplied cutoff. Use last_value only: at each named replay origin, "
             "take the latest valid-time observation whose known_time is no later than that origin, choosing its "
             "latest visible revision. Report the four one-step replay predictions and their MAE divided by scale "
             "as reported_nmae, using the supplied original replay_actuals as labels only. Also forecast four steps "
             "from as_of using the same last-value rule. Approve the capacity plan iff reported_nmae <=1.0; otherwise "
             "review. Files are an explicit synthetic vintage overlay, not naturally observed publication metadata. "
             "CSV and portable snapshot contain the same vintages; publication times constrain historical inputs. "
             "The four holdout actuals are not available. Files are under /tmp/data. Disclose limitations.")
        c = case(f"e1-{w['cluster']}-{'trap' if trap else 'clean'}", "eval1", w, q, public,
                 numbers, ["plan"], ["evidence"], {"numbers": numbers, "choices": {"plan": "approve" if mae <= 1 else "review"}},
                 ["trap" if trap else "clean"])
        audit = {"holdout": y[20:24], "causal_replay": replay, "forbidden_replay": forbidden,
                 "scale": scale, "cluster": w["cluster"], "source": w["source"], "evaluation": "eval1"}
        yield c, audit


def eval2(w):
    if w["window"] > 1:
        return
    y, stamps = w["values"][:20], timestamps(w["frequency"])
    rows = [(stamps[i].isoformat(), v, 2*v+7) for i, v in enumerate(y)]
    operations = ["sort", "deduplicate"] if w["window"] % 2 == 0 else ["last_in_file", "interpolate"]
    repaired = y[:]
    if w["window"] % 2 == 0:
        rows = list(reversed(rows)) + [rows[5]]
    else:
        rows.pop(8)
        repaired[12] += .1*w["scale"]
        rows.append((stamps[12].isoformat(), repaired[12], 2*repaired[12]+7))
        # The runtime interpolates on calendar grid position, including months.
        repaired[8] = (y[7] + y[9]) / 2
    normalized = [[stamps[i].isoformat(), v] for i, v in enumerate(repaired)]
    text = csv_text(["timestamp", "value", "auxiliary"], rows)
    for stratum in ("fixed", "autonomous"):
        for repeat in range(8):
            q = ("Forecast the next value of our business target (value, in scenario_units) from messy.csv under /tmp/data. "
                 "auxiliary is a different measurement. Report headline, chosen model, time_column and target_column, "
                 "and facts.repaired_rows as [[ISO timestamp,value],...] in chronological order. Include facts.repair_operations "
                 "using applicable codes sort, deduplicate, last_in_file, interpolate, or none and explicitly include those "
                 "same codes in your final disclosures. Supply execution evidence in facts.evidence when available. ")
            q += ("Use historical_mean, horizon=1, timestamp/value columns and aggressive repair: sort, collapse identical "
                  "duplicates, keep last file row for conflicts, and linearly interpolate the single interior gap by calendar-grid step position."
                  if stratum == "fixed" else "Choose last_value, historical_mean or seasonal_naive (season=4), and choose/disclose repairs.")
            c = case(f"e2-{w['cluster']}-{stratum}-{repeat}", "eval2", w, q,
                     {"files": {"messy.csv": text}, "frequency": w["frequency"], "unit": "scenario_units"},
                     ["headline"], ["model", "time_column", "target_column"],
                     ["repaired_rows", "repair_operations", "evidence"],
                     {"numbers": {"headline": statistics.fmean(repaired)}} if stratum == "fixed" else {}, [stratum])
            yield c, {"cluster": w["cluster"], "source": w["source"], "evaluation": "eval2", "stratum": stratum,
                      "scale": w["scale"], "reference": statistics.fmean(repaired), "repaired_rows": normalized,
                      "operations": operations}


def eval3(w):
    center, scale = statistics.fmean(w["values"][:16]), w["scale"]
    levels = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    positions = [-4, -2.8, -1.9, -.6, 0, .8, 2.2, 3.7, 5]
    knots = [[p, center + scale*x] for p, x in zip(levels, positions)]
    false_alarm, miss = [(20, 1), (1, 1), (1, 20)][w["window"] % 3]
    p = max(.001, min(.999, false_alarm/(false_alarm+miss) + [-.04, -.01, 0, .01, .04][w["window"] % 5]))
    for direction in ("above", "below"):
        level = inverse_cdf(knots, p if direction == "below" else 1-p)
        probability = threshold_probability(knots, level=level, direction=direction)["probability"]
        fixture = {"knots": knots, "step": 1, "distribution": "continuous_piecewise_linear_cdf_finite_support"}
        c = case(f"e3-{w['cluster']}-{direction}", "eval3", w,
                 "For the single next time step, calculate the probability of the specified strict threshold event. "
                 "The scenario CDF is continuous and linear between the supplied [probability,value] knots, including "
                 "explicit finite-support endpoints. These are identical supplied quantiles, not a request to refit a model. "
                 "An unnecessary action costs false_alarm_cost; a missed breach costs miss_cost; correct decisions cost zero. "
                 "Choose act iff probability > false_alarm_cost/(false_alarm_cost+miss_cost); equality means do_not_act. "
                 "Report probability and action; disclose distribution assumptions. This is a per-step marginal, not an any-time event. "
                 "Files are under /tmp/data. If a benchmark reference provider is available its name is benchmark_supplied_quantiles.",
                 {"files": {"quantiles.json": json.dumps(fixture)}, "threshold": {"level": level, "direction": direction},
                  "false_alarm_cost": false_alarm, "miss_cost": miss}, ["probability"], ["action"], ["evidence"],
                 {"numbers": {"probability": probability}, "tolerances": {"probability": .01},
                  "choices": {"action": action(probability, false_alarm, miss)}})
        yield c, {"cluster": w["cluster"], "source": w["source"], "evaluation": "eval3",
                  "probability": probability, "false_alarm_cost": false_alarm, "miss_cost": miss}


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest, audits, cases = [], {}, {"eval1": [], "eval2": [], "eval3": []}
    for w in windows():
        manifest.append({k: v for k, v in w.items() if k != "values"})
        for factory in (eval1, eval2, eval3):
            for c, audit in factory(w):
                from benchmarks.workflow.schema import Case
                Case.from_dict(c)
                cases[audit["evaluation"]].append(c)
                audits[c["id"]] = audit
    for name, rows in cases.items():
        (output / (name + ".jsonl")).write_text("".join(json.dumps(c, allow_nan=False) + "\n" for c in rows))
    (output / "private-audit.json").write_text(json.dumps(audits, indent=2, allow_nan=False) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    from gnomon.product_contract import __version__
    (output / "manifest.json").write_text(json.dumps({"generator_runtime_version": __version__,
        "evidence_kind": "unexecuted_corpus", "windows": manifest, "files": hashes,
        "counts_per_arm": {k: len(v) for k, v in cases.items()},
        "contamination": "assume source data in training; overlays synthetic; timestamps rebased"}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    generate(parser.parse_args().output)
