"""Independent receipt/visibility audit; no forecasts, writes or aggregate scores."""

import argparse, hashlib, json, math, sqlite3
from datetime import datetime
from pathlib import Path
from statistics import mean
from gnomon.final_selection import forecast_request_fingerprint


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def instant(s):
    return datetime.fromisoformat(s)


def audit(prepared, output, allow_development=False):
    cases = json.loads((prepared / "cases.json").read_text())
    memory_path = prepared / "memory/memory.json"
    if not memory_path.exists():
        raise ValueError("Prepared memory missing")
    memory = json.loads(memory_path.read_text())
    if not allow_development:
        run = prepared.parent / "confirmation-agent-010"
        completed = list(run.glob("[0-9]*.json"))
        if len(completed) != 3744 or list(run.glob("*.harness-error.json")):
            raise ValueError("Confirmation must finish before this audit")
    db = prepared / "memory/ledger.db"
    before = digest(db)
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    actuals = {r["actual_id"]: dict(r) for r in conn.execute("SELECT * FROM actuals")}
    executions = {}
    for r in conn.execute(
        "SELECT e.*, p.payload_json FROM executions e JOIN payloads p USING(payload_id)"
    ):
        if hashlib.sha256(r["payload_json"].encode()).hexdigest() != r["payload_id"]:
            raise ValueError("Payload digest mismatch")
        executions[r["execution_id"]] = {
            **json.loads(r["payload_json"]),
            "recorded_at": r["recorded_at"],
        }
    packets = {(p["series_id"], p["round"]): p for p in memory["packets"]}
    checks = {
        "cases": 0,
        "executions": 0,
        "historical_origin_exposures": 0,
        "actual_visibility_checks": 0,
        "raw_score_checks": 0,
    }
    for c in cases:
        now = instant(c["origin"])
        p = packets[c["series_id"], c["round"]]
        requests = []
        for eid in c["execution_ids"]:
            e = executions[eid]
            req = e["request"]
            requests.append(req)
            assert req["series_id"] == c["series_id"] and req["horizon"] == len(
                c["actual"]
            )
            assert (
                len(req["timestamps"]) == len(req["history"])
                and max(map(instant, req["timestamps"])) == now
            )
            assert (
                list(req["future_timestamps"]) == c["future_timestamps"]
                and min(map(instant, req["future_timestamps"])) > now
            )
            assert instant(e["recorded_at"]) == now
            assert forecast_request_fingerprint(req) == p["request_fingerprint"]
            checks["executions"] += 1
        prior = {
            instant(x["origin"]): x
            for x in cases
            if x["series_id"] == c["series_id"]
            and instant(x["origin"]) < now
            and instant(x["outcome_recorded_at"]) <= now
            and max(map(instant, x["future_timestamps"])) <= now
        }
        raw = p["raw_history"]
        got = [instant(r["origin"]) for r in raw["records"]]
        assert len(got) == len(set(got)) and set(got) == set(prior)
        assert (
            instant(raw["source_as_of"]) == now
            and instant(raw["recorded_as_of"]) == now
        )
        for row in raw["records"]:
            old = prior[instant(row["origin"])]
            for i, name in enumerate(raw["providers"]):
                point = old["predictions"][name]
                actual = old["actual"]
                expected_mae = mean(
                    abs(v - a) for v, a in zip(point, actual, strict=True)
                )
                expected_rmsle = math.sqrt(
                    mean(
                        (math.log1p(max(v, 0)) - math.log1p(a)) ** 2
                        for v, a in zip(point, actual, strict=True)
                    )
                )
                assert math.isclose(
                    expected_mae, row["mae"][i], rel_tol=1e-12, abs_tol=1e-12
                )
                assert math.isclose(
                    expected_rmsle, row["rmsle"][i], rel_tol=1e-12, abs_tol=1e-12
                )
                checks["raw_score_checks"] += 2
        evidence = json.loads(
            (
                prepared / "memory" / f"{c['round']:04d}-{c['series_id']}.evidence.json"
            ).read_text()
        )
        comparison = evidence["unfiltered_comparison"]
        assert set(instant(o["origin"]) for o in comparison["origins"]) == set(prior)
        for origin in comparison["origins"]:
            old = prior[instant(origin["origin"])]
            assert len(origin["actual_ids"]) == len(old["actual"])
            valid = []
            for aid in origin["actual_ids"]:
                a = actuals[aid]
                assert (
                    a["series_id"] == c["series_id"]
                    and a["unit"] == requests[0]["unit"]
                )
                assert (
                    instant(a["recorded_at"]) <= now
                    and instant(a["source_available_at"]) <= now
                )
                valid.append(instant(a["valid_time"]))
                checks["actual_visibility_checks"] += 1
            assert set(valid) == set(map(instant, old["future_timestamps"]))
            checks["historical_origin_exposures"] += 1
        checks["cases"] += 1
    conn.close()
    assert digest(db) == before
    result = {
        "scope": memory["scope"],
        "checks": checks,
        "all_checks_passed": True,
        "ledger_unchanged": True,
        "ledger_sha256": before,
        "cases_sha256": digest(prepared / "cases.json"),
        "memory_sha256": digest(memory_path),
        "provider_calls": 0,
        "target_established": False,
        "limitation": "Validates recorded synthetic replay facts and numerical history. Does not establish real-world historical availability or broad leakage safety.",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("prepared", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-development", action="store_true")
    a = p.parse_args()
    audit(a.prepared, a.output, a.allow_development)
