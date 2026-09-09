# Rescore revised actuals without moving forecast origins

Run this synthetic example in a new scratch directory. It uses only installed
public Python APIs and fixed test clocks. In production, ingest with the real
recording clock and accurate source availability timestamps.

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path
from gnomon import GnomonSession, InferenceEngine, TemporalLedger
from gnomon.temporal_store import TemporalStore, TemporalObservation
from gnomon.ids import FixedClock

def at(day):
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)

store = TemporalStore(Path('vintages.db'))
for day in range(1, 31):
    store.ingest_rows('sales', [TemporalObservation(
        entity='sales', variable='value', valid_time=at(day),
        known_time=at(day), value=float(day))],
        source_fingerprint=f'original-{day}', clock=FixedClock(at(day)))

# The configuration loads the three built-ins. Clock 35 is after all original
# observations and before the later revisions, keeping the example reproducible.
Path('providers.toml').write_text('schema_version=1\nledger_path="evidence.db"\n')
with GnomonSession.from_config('providers.toml') as session:
    session.ledger.clock = FixedClock(at(35))
    inspected = session.data.inspect('store:sales', store_path='vintages.db',
        unit='widgets', as_of=at(30).isoformat(), recorded_as_of=at(30).isoformat())
    original = session.evaluate(inspected['data_ref'], candidates=['historical_mean'],
        baseline='last_value', horizon=2, folds=4, verify=True)
    original_id = original['study_id']
    saved_original = session.ledger.study(original_id)

# These observations apply to January dates, became source-available on day 40,
# and were recorded locally on day 42. They do not change what the providers saw.
for fold in saved_original['folds']:
    for actual in fold['actuals']:
        store.ingest_rows('sales', [TemporalObservation(
            entity='sales', variable='value',
            valid_time=datetime.fromisoformat(actual['valid_time']),
            known_time=at(40), value=fold['request']['history'][-1])],
            source_fingerprint='revised', clock=FixedClock(at(42)))

# Rescoring works with an empty provider registry: predictions come from ledger
# executions, so no custom provider imports or model calls are required.
ledger = TemporalLedger('evidence.db', clock=FixedClock(at(45)))
with GnomonSession(InferenceEngine(ledger=ledger)) as session:
    revised = session.data.inspect('store:sales', store_path='vintages.db',
        unit='widgets', as_of=at(45).isoformat(), recorded_as_of=at(45).isoformat())
    before_recording = session.rescore(revised['data_ref'], study_id=original_id,
        source_as_of=at(45).isoformat(), recorded_as_of=at(41).isoformat())
    assert before_recording['scores'] == saved_original['scores']
    rescored = session.rescore(revised['data_ref'], study_id=original_id,
        source_as_of=at(45).isoformat(), recorded_as_of=at(45).isoformat(),
        allow_partial=False)
    comparison = session.compare_studies(original_study_id=original_id,
        rescored_study_id=rescored['study_id'])
    assert rescored['provider_calls'] == comparison['provider_calls'] == 0
    assert comparison['predictions_reused_exactly']
    assert comparison['original_unchanged']
    assert session.ledger.study(original_id) == saved_original
    assert rescored['scores']['last_value']['mae'] == 0
    assert rescored['score_derivations']['last_value']['absolute_error_sum'] == 0
    print(comparison)
```

The `clock` in this example is a public ledger constructor argument. Prefer
constructing the ledger with the intended clock when integrating test fixtures;
ordinary applications should leave it at its real clock.

Equivalent CLI operations after preparing original evidence are:

```sh
gnomon evaluate --rescore ORIGINAL_STUDY_ID --input store:sales \
  --store-path vintages.db --unit widgets --ledger-path evidence.db \
  --source-as-of 2026-02-14T00:00:00Z --recorded-as-of 2026-02-14T00:00:00Z \
  --no-partial --save-result rescored.json
gnomon evaluate --compare ORIGINAL_STUDY_ID RESCORED_STUDY_ID --ledger-path evidence.db
```

IDs above are placeholders; use returned study IDs. The MCP equivalents are
`gnomon_evaluate` with `operation: rescore`, `study_id`, `data_ref` and both
cutoffs, then `operation: compare_studies`, `original_study_id` and
`rescored_study_id`. A request containing only `study_id` **retrieves** saved full
evidence; it never refreshes scores.

Rescore source/recording cutoffs filter actual availability. The original
`fold.origin`, `fold.request`, predictions and valid-time forecast grid stay
unchanged. A previously frozen snapshot may be narrower than a requested cutoff;
inspect the store again to access later vintages. Output `cutoff_scopes` discloses
both requested and effective boundaries.

`allow_partial` defaults true. Missing original predictions or unavailable
actuals exclude a fold, producing partial/unscored evidence. `--no-partial`
rejects instead, without saving a rescore. Revised plain CSV data cannot establish
revision availability; use explicit store vintages. Rescoring is a new immutable
study, with original/rescore IDs and the original payload hash. Cohort IDs may
change because actual vintages or scored folds changed. Comparisons report old/new
metrics, rankings, ties, changed actual identities and exact prediction reuse.

Routing remains a prospective selection operation: its input and source cutoff
must also define a future forecast grid. Use rescore when the goal is to update
historical scores without defining a new forecast origin.
