# Agent operation discovery

Use `gnomon schemas` for the schema index. `gnomon forecast` and `gnomon infer`
are aliases. `--input` forecasts require a provider and horizon; routing from
input requires a study plus source and recording cutoffs.

| Python construction | Registered providers |
| --- | --- |
| `GnomonSession.from_config()` | Offline built-ins |
| `GnomonSession.from_config('providers.toml')` | Built-ins plus explicit operator configuration |
| `GnomonSession()` | Empty; explicit custom registry |
| `InferenceEngine()` | Empty; use `register` / `register_factory` |

`gnomon providers example` prints an installed, runnable, unit-bearing custom
provider. Run it with `python -m gnomon.examples.custom_provider`. Providers return
`ForecastResult`, echoing series/unit and future instants. `result.point` is a
tuple field; `result.points()` returns a list copy. Execution `to_dict()` includes
cache policy, request provenance and the effective season.

## Configuration and cache

Relative TOML paths resolve against the **configuration file directory**, not
the working directory. Explicit CLI paths resolve against the working directory.
Inspect configuration without importing providers, opening a ledger, or resolving
secrets:

```sh
gnomon capabilities --providers-config providers.toml --show-resolved-config
gnomon capabilities --config-schema
gnomon capabilities --cache
gnomon capabilities --providers-config providers.toml --cache \
  --provider last_value --request '{"history":[1,2,3],"horizon":2}'
```

Resolved inspection shows config location, effective ledger path and existence,
cache size, write permissions, temporal availability and provider entrypoints.
It does not import/check provider modules or contact remote services. Cache
diagnostics start the configured session but do not forecast; configured provider
initialization/discovery can still run, as in ordinary startup.

Cache keys use validated canonical requests. Integral floats and equal integers,
lists/tuples, explicit empty/default arrays and equivalent ISO instants share
identity. Large integers stay exact. Without a calendar frequency, aware timestamps become UTC ISO strings;
naive timestamps remain naive. Unit and series labels stay exact. Invalid booleans,
numeric strings, malformed shapes and nonfinite values are rejected before hashing.
Calendar-frequency requests retain local offsets because a daily DST step may
span 23 or 25 elapsed hours; those calendar identities are not merged with UTC
grids. Canonical input is what providers receive and the ledger records.

`engine.cache_diagnostic(provider, request)` returns the canonical request,
fingerprint, eligibility and lookup reason without executing the provider. A
single persistent engine can report a hit; independent CLI processes cannot share
entries. A missing entry can result from first use or eviction. A provider needs
an explicit revision and deterministic registration to be eligible.

## Completion, verification and recovery

Capabilities publishes `exit_semantics` and `cutoff_semantics`. Exit 0 means an
operation executed; it does not establish complete scoring or an evidence-based
recommendation. Exit 2 covers rejection/unscored/invalid requests; exit 3 covers
partial evaluation; exit 130 denotes interruption. Response completion fields
describe the requested technical operation, not whether its business assumptions
were correct. `evidence_complete: null` means completeness of scoring evidence is
not applicable. Forecasting alone never establishes scored evidence.

`effective_season` is explicit in forecasts. `season_defaulted` identifies omitted
CLI/MCP periods. A seasonal period of 1 repeats the last value.

`forecast --verify` returns an independent built-in calculation and expected
points. `evaluate --verify` adds scored-pair hashes, error vectors, metric sums and
denominators. These derivations help audit arithmetic; they do not demonstrate
forecast superiority or independent real-world truth.

Every error exposes a common `error.recovery` object with supplied/preserved/
changed/rejected fields, choices, example kind, runnable flag and admissibility.
Null admissibility means unestablished. Inspect the referenced example and its
guidance; a runnable illustration may still contain choices the user must make.
CLI input errors distinguish explicit options from defaults. Frozen `.gnomon`
inputs reject preparation flags, including identical explicit values; return to
the original source to prepare a different snapshot.

Use `gnomon --compact-errors COMMAND ...` or MCP operator `compact_errors = true`
to replace the legacy duplicate `rejection` body with an `/error` reference.
Existing clients retain their original default shape.

Repair errors provide separate drop, gap-fill, conflict-resolution and timestamp-
alignment budget scopes. Null counts were not measured by that diagnostic.
Fills and conflicts share one 30% original-observation budget; drops have their
own 5% all-row budget. A bounded scan never presents an incomplete count as exact.
Mixed invalid timestamp/target cells count as two bad fields on one affected row.
Safe mode never drops unparseable rows, even when aggressive's budget permits it.

`gnomon inspect --diagnose --input data.csv --frequency D` performs a bounded
dry run of all three repair modes before choosing one. It does not change the
source, write repaired output, call providers or return a reusable reference.
The equivalent MCP call is `gnomon_inspect` with `diagnose: true`. Inputs are
limited to local files up to 8 MiB and 100,000 retained observations. Preparation
success is not provider suitability; stages after an early rejection remain
unmeasured and lower-bound counts stay labelled.

## Self-check coverage

```sh
gnomon self-check leakage --cases 20 --seed 7 --families snapshot \
  revision_visibility recorded_visibility fold_boundaries future_covariates \
  multiple_series dst repaired_histories cache_equivalence
```

The result lists exercised families/providers and scope. Covariate checks cover
shape/capability validation, not genuine future publication availability. Multiple
series checks cover store isolation, not arbitrary panel models. DST checks need
host IANA timezone data. These are finite reproducible mechanism checks, not a
general proof of leakage safety or client/provider compatibility.

`gnomon --version` prints the build-qualified runtime ID; distribution metadata
reports the release version. Capabilities explains both and exposes interface
availability: routing/ledger MCP tools appear when a ledger is configured.

Continue with the [local workflow](local-evidence-workflow.md),
[revised-vintage workflow](revised-vintage-workflow.md), and
[MCP example](mcp-evidence-workflow.md).

## Follow-up discovery and result contracts

Use `gnomon capabilities --brief` (or Python `capabilities(brief=True)` / MCP
`gnomon_capabilities` with `{"brief":true}`) for shared request schemas instead
of repeating them per provider. Use `describe --brief` to omit the repeated
inspection object. Normal per-command help remains self-contained.

`gnomon providers example --raw` prints executable Python; `--write-to FILE.py`
creates a new source file and refuses to overwrite one. Live cache statistics
are `entries`, `hits`, `misses` and `evictions`. Hits/misses count only eligible
lookups; diagnostics, disabled caching, bypasses and invalid requests do not
increment them. Counters cover the engine lifetime, not other CLI processes.

`gnomon evaluate --input data.csv --candidates historical_mean --baseline
last_value --horizon 2 --preflight` reports planned origins, replay mode and
visibility without forecasts or a saved study. Use `--replay recorded` or
`--replay source_available` only after choosing the appropriate temporal
question. Inspection discloses the default and points to preflight; it cannot
know the requested horizons and origins in advance.

Default route fallback keeps exit 0 and emits one stderr warning. Use
`--require-evidence` for exit 2 on fallback. `--save-result` saves both success
and rejection JSON; if saving fails, stdout retains the original rejection and
names the unsaved path.

| Result surface | Point forecast location |
| --- | --- |
| CLI/MCP inference result | `result.point` |
| Python ForecastExecution | `result.point` tuple; `result.points()` list-copy method |
| Saved study fold | `folds[i].runs[provider].point` |

Study runs are compact evidence records, not ForecastExecution objects. Their
existing shape remains stable. Completed backtest runs appear in ledger search
as `scored_in_study`, with their saved study IDs. These are immutable study
scores, not production-ledger actuals rescored at the search source cutoff.

Completion fields have separate scopes:

- `operation_succeeded`: the interface operation executed.
- `task_completed`: the technical task completed; a default routing fallback
  or incomplete requested scoring is false. A history query can complete while
  finding insufficient evidence.
- `scoring_complete`: whether requested scoring is complete; null when no
  scoring operation was requested.
- `evidence_status`: `complete`, `insufficient`, or `not_applicable` for the
  scoring/selection evidence. The older `evidence_complete` boolean/null stays
  available for compatibility; null means not applicable here.
- `returned_evidence`: `complete_payload`, `fold_summary`, or `partial_payload`.
  A fold summary can describe complete scoring without including all fold
  vectors. Follow `full_study` for those vectors; follow `full_result` for a
  retained response payload.

The response byte budget counts compact UTF-8 encoded structured payload. It
excludes JSON-RPC framing, duplicated MCP text blocks, transport headers and
pretty-printed CLI whitespace. Retained-result hashes and character counts apply
to the exact root text returned by `gnomon_read`.

`gnomon self-check families` describes each fixture, assertion and limitation.
Add `--detailed` to a leakage run for expected and actual values per case.
See the [prospective comparison example](production-history-comparison.md) for
built-ins paired with a controlled-clock TemporalLedger.
