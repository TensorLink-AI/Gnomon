# Local evidence workflow

Run this in a fresh scratch directory after installing Gnomon into the active
Python environment. It uses synthetic daily observations, local files and the
three offline built-ins. The TOML explicitly enables writes of these synthetic
actuals. Use real source availability times, units and cutoffs for your own data.

```bash
set -eu
python - <<'PY'
from datetime import date, timedelta
from pathlib import Path
rows = [f"{date(2026, 1, 1) + timedelta(days=i)},{i % 7 + 1}" for i in range(28)]
Path("data.csv").write_text("timestamp,value\n" + "\n".join(rows) + "\n")
Path("writer.toml").write_text('schema_version = 1\nledger_path = "evidence.db"\nallow_outcome_writes = true\n')
PY

gnomon inspect --input data.csv --timezone UTC --unit widgets \
  --as-of 2026-01-28T00:00:00Z --save-snapshot data.gnomon

gnomon forecast --input data.gnomon --provider seasonal_naive --season 7 \
  --horizon 2 --ledger-path evidence.db --save-result forecast.json

gnomon evaluate --input data.gnomon --baseline last_value \
  --candidates seasonal_naive historical_mean --season 7 --horizon 2 --folds 4 \
  --ledger-path evidence.db --save-result study.json

study_id="$(python -c 'import json; print(json.load(open("study.json"))["study_id"])')"
recording_cutoff="$(python -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat())')"
gnomon route --input data.gnomon --study "$study_id" --ledger-path evidence.db \
  --source-as-of 2026-01-28T00:00:00Z --recorded-as-of "$recording_cutoff" \
  --save-result route.json

gnomon ledger --providers-config writer.toml --arguments '{"operation":"append_actual","actuals":[{"series_id":"__default__","unit":"widgets","valid_time":"2026-01-29T00:00:00Z","source_available_at":"2026-02-01T00:00:00Z","value":1},{"series_id":"__default__","unit":"widgets","valid_time":"2026-01-30T00:00:00Z","source_available_at":"2026-02-01T00:00:00Z","value":2}]}'

python - <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
execution_id = json.loads(Path("forecast.json").read_text())["execution_id"]
Path("score-request.json").write_text(json.dumps({
    "operation": "evaluate", "execution_id": execution_id, "allow_partial": False,
    "source_as_of": "2026-02-01T00:00:00Z",
    "recorded_as_of": datetime.now(timezone.utc).isoformat(),
}))
Path("history-request.json").write_text(json.dumps({"operation": "evaluations", "execution_id": execution_id}))
PY

gnomon ledger --ledger-path evidence.db --arguments @score-request.json --save-result score.json
gnomon ledger --ledger-path evidence.db --arguments @score-request.json --save-result retry.json
gnomon ledger --ledger-path evidence.db --arguments @history-request.json --save-result history.json
```

Check `study.json` for complete evaluation and `routing_readiness.ready` before
routing. `route.json` should select seasonal naive from matched evidence with
`fallback_used: false` and zero provider calls. This synthetic result makes no
claim about future predictive superiority.

The forecast is [1, 2] widgets. The matching actuals give a complete zero-error
score. `score.json` and `retry.json` have the same evaluation ID; the second has
`evaluation_reused: true`. Saved `coverage` describes the original score;
`current_coverage` describes the repeated query. The retained request file fixes
both cutoffs, so the retry does not silently advance the evidence boundary.
Historical retrieval reports the original saved evidence.

For persistent Python caching, including a unit-bearing request, see the
[Python example](python-api.md). `gnomon schemas` lists all schemas; the
[scoring and recovery contract](scoring-and-recovery.md) explains partial scores,
unit defaults, repair choices and bounded response retrieval.
