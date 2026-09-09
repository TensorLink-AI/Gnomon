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
