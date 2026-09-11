#!/usr/bin/env bash
# The agent-cheating demo: source cutoffs, recorded replay and revision-aware
# rescoring, offline, with the three built-in baselines. No key, no network.
#
#   bash examples/agent_cheating_demo.sh
#
# Every step is the same CLI/Python surface an agent uses. Nothing here is a
# claim about forecast accuracy; it shows what Gnomon refuses to let a model
# see, and what it keeps when the actuals later change.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CSV="$HERE/messy_requests_revisions.csv"
if [ -n "${GNOMON_DEMO_WORK:-}" ]; then
  WORK="$GNOMON_DEMO_WORK"; mkdir -p "$WORK"   # kept for inspection
else
  WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
fi
STORE="$WORK/vintages.db"
LEDGER="$WORK/evidence.db"
JQ="gnomon python -"

say() { printf '\n== %s\n' "$*"; }

say "0. Load the revision history: every row is its own vintage, recorded the day it was published"
$JQ "$STORE" "$CSV" <<'PY'
import csv, sys
from datetime import datetime, timezone
from pathlib import Path
from gnomon.ids import FixedClock
from gnomon.temporal_store import TemporalObservation, TemporalStore
store_path, csv_path = sys.argv[1:3]
store = TemporalStore(Path(store_path))
def day(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
rows = list(csv.DictReader(open(csv_path, newline="")))
by_published = {}
for row in rows:
    by_published.setdefault(row["published"], []).append(TemporalObservation(
        entity="__default__", variable="requests", valid_time=day(row["timestamp"]),
        known_time=day(row["published"]), value=float(row["requests"])))
added = revised = 0
for published in sorted(by_published):
    report = store.ingest_rows("requests", by_published[published], source_fingerprint=f"published:{published}",
                               clock=FixedClock(day(published)))
    added += report.rows_added; revised += report.revisions_created
print(f"vintages: {added}  revisions of an earlier value: {revised}")
PY

say "1. Inspect as of 2026-05-01: later vintages are excluded before any model runs"
gnomon inspect store:requests --store-path "$STORE" --target requests \
  --as-of 2026-05-01T00:00:00+00:00 --for evaluate > "$WORK/inspect.json"
$JQ "$WORK/inspect.json" <<'PY'
import json, sys
snapshot = json.load(open(sys.argv[1]))["snapshot"]
print(f"snapshot_id: {snapshot['snapshot_id']}  as_of: {snapshot['as_of']}  known_time_provenance: {snapshot['known_time_provenance']}")
PY
$JQ "$STORE" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path
from gnomon.temporal_store import TemporalStore
visibility = TemporalStore(Path(sys.argv[1])).snapshot("requests", datetime(2026, 5, 1, tzinfo=timezone.utc)).visibility
print(f"input_vintages: {visibility['input_vintages']}  visible: {visibility['visible_vintages']}  "
      f"excluded_by_source_cutoff: {visibility['excluded_by_source_cutoff']}")
PY

say "2. Evaluate with recorded replay at 2026-06-01: each fold sees only what was recorded by its origin"
gnomon evaluate --input store:requests --store-path "$STORE" --target requests \
  --as-of 2026-06-01T00:00:00+00:00 --recorded-as-of 2026-06-01T00:00:00+00:00 --replay recorded \
  --candidates seasonal_naive historical_mean --baseline last_value --season 7 --horizon 7 --folds 3 \
  --ledger-path "$LEDGER" --json > "$WORK/study.json"
$JQ "$WORK/study.json" <<'PY'
import json, sys
study = json.load(open(sys.argv[1]))
print(f"study_id: {study['study_id']}  status: {study['status']}  replay: {study['replay']}")
for fold in study["folds"]:
    v = fold["visibility"]
    print(f"  origin {fold['origin'][:10]}: excluded_by_source_cutoff={v['excluded_by_source_cutoff']} "
          f"excluded_by_recorded_cutoff={v['excluded_by_recorded_cutoff']} status={fold['status']}")
print("  scores (mae): " + ", ".join(f"{p}={s['mae']:.3f}" for p, s in study["scores"].items()))
print(f"  ranking: {' > '.join(study['ranking'])}")
PY
STUDY_ID="$($JQ "$WORK/study.json" <<'PY'
import json, sys; print(json.load(open(sys.argv[1]))["study_id"])
PY
)"

say "3. A revised actual for 2026-05-25 arrives on 2026-06-13 (the original prediction is already recorded)"
$JQ "$STORE" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path
from gnomon.ids import FixedClock
from gnomon.temporal_store import TemporalObservation, TemporalStore
store = TemporalStore(Path(sys.argv[1]))
when = datetime(2026, 6, 13, tzinfo=timezone.utc)
report = store.ingest_rows("requests", [TemporalObservation(
    entity="__default__", variable="requests", valid_time=datetime(2026, 5, 25, tzinfo=timezone.utc),
    known_time=when, value=299.0)], source_fingerprint="late-correction", clock=FixedClock(when))
print(f"rows_added: {report.rows_added}  revisions_created: {report.revisions_created}")
PY

say "4. Rescore the same study with sources as of 2026-06-14: predictions reused exactly, scores refreshed, originals untouched"
# The recording cutoff bounds locally recorded evidence (the study was recorded
# just now, on the real clock), so 2099 means everything recorded so far.
gnomon evaluate --rescore "$STUDY_ID" --input store:requests --store-path "$STORE" --target requests \
  --source-as-of 2026-06-14T00:00:00+00:00 --recorded-as-of 2099-01-01T00:00:00+00:00 \
  --ledger-path "$LEDGER" --json > "$WORK/rescored.json"
RESCORED_ID="$($JQ "$WORK/rescored.json" <<'PY'
import json, sys; print(json.load(open(sys.argv[1]))["study_id"])
PY
)"
gnomon evaluate --compare "$STUDY_ID" "$RESCORED_ID" --ledger-path "$LEDGER" --json > "$WORK/compare.json"
$JQ "$WORK/compare.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(f"provider_calls: {c['provider_calls']}  predictions_reused_exactly: {c['predictions_reused_exactly']}  "
      f"original_unchanged: {c['original_unchanged']}")
print(f"changed actuals: {len(c['changed_actual_ids'])} fold(s)")
for provider in c["old_scores"]:
    print(f"  {provider}: mae {c['old_scores'][provider]['mae']:.3f} -> {c['new_scores'][provider]['mae']:.3f}")
print(f"  ranking: {' > '.join(c['old_ranking'])} -> {' > '.join(c['new_ranking'])}")
PY

say "done: original study $STUDY_ID kept; rescore $RESCORED_ID appended"
