# Covariate enrichment

Gnomon can evaluate externally sourced, future-known variables such as published
holiday calendars, scheduled prices, or archived weather-forecast vintages.
The agent decides what might matter and fetches it. Gnomon owns the temporal data
contract and decides whether the proposal improves rolling backtests.

MCP is optional. The same workflow is available through the CLI and Python API.

## Point-in-time data contract

Each row needs two timestamps:

- `timestamp`: when the value applies (`valid_at`)
- `known_at`: when that exact value or forecast vintage became available

```csv
timestamp,known_at,is_holiday,temperature_forecast
2026-08-01T00:00:00+00:00,2026-01-01T00:00:00+00:00,0,28.5
2026-08-02T00:00:00+00:00,2026-01-01T00:00:00+00:00,0,27.3
```

Multiple rows may share `timestamp`. This represents revised vintages. At a
historical fold cutoff, Gnomon selects the most recent row whose `known_at` is no
later than the cutoff.

Do not label a reconstructed historical observation as a forecast. For example,
today's archive of observed temperatures does not reveal what weather forecast
was available six months ago. Weather inputs require archived issued forecasts.
If those vintages are unavailable, Gnomon rejects the proposal for backtesting.

This release accepts numeric `continuous`, `binary`, and declared cyclic
features (`cyclic_<positive-period>`, such as `cyclic_1440` for minute of day)
whose
availability is explicitly `future_known`. It does not silently encode
categoricals, interpolate gaps, or fetch URLs.

## CLI workflow

Get the exact forecast timestamps and selection-fold cutoffs:

```bash
gnomon covariates guide sales.csv \
  --time timestamp --target sales --frequency D --horizon 14
```

Validate a locally fetched proposal:

```bash
gnomon covariates validate sales.csv \
  --time timestamp --target sales --frequency D --horizon 14 \
  --covariates covariates.csv \
  --covariate-mapping 'is_holiday:binary:future_known,temperature_forecast:continuous:future_known'
```

Evaluate it and produce a forecast:

```bash
gnomon forecast sales.csv \
  --time timestamp --target sales --frequency D --horizon 14 \
  --covariates covariates.csv \
  --covariate-mapping 'is_holiday:binary:future_known,temperature_forecast:continuous:future_known'
```

For panel data, pass both `--series` for the target file and
`--covariate-series` for the covariate file.

## Admission procedure

Gnomon always retains the evaluated univariate forecast as the control. Candidate
features enter in declared order through forward selection. A feature is kept
only when it:

1. is available at every selection cutoff and corresponding horizon;
2. meets the configured mean improvement threshold;
3. improves a majority of selection folds; and
4. is not supported by only one anomalous fold.

The calibration fold estimates residual intervals. The final test fold only
reports coverage and never controls feature admission. The final forecast uses
only features admitted on selection folds.

The artifact records retained and rejected features, fold improvements, source
path, content fingerprint, and measured test coverage in `evidence.jsonl`.

## Combining covariates with context events

A run may supply both `--covariates` and `--context`. Each enrichment first
faces its own independent ablation gate exactly as in a single-enrichment
run. An adjudication stage then compares the base model and every admitted
challenger — base + context, base + covariates, base + both — on identical
selection folds (same origins, same cutoffs), and picks the winner
deterministically: best mean fold score, ties broken by fewest enrichments,
then by fixed candidate order. The combined candidate is the covariate
linear forecast plus the additive event effect, fitted per fold under that
fold's cutoff; when it wins, the result reports
`selected_model: "combined_enrichment"`. The whole comparison — candidates,
per-fold scores, winner, and reason — is recorded as an
`enrichment_adjudication` evidence record, so the artifact proves the
choice rather than asserting it.

## Agent and MCP workflow

Agents pass covariates directly to `gnomon_forecast`, either inline or as a
local file, and may call `gnomon_validate_covariates` first. Human operators
can use `gnomon covariates guide|validate` from the CLI.

An agent should use its own permitted discovery tools to retrieve data and
preserve its source URL separately; Gnomon will not dereference arbitrary
URLs. Feature admission always runs inside the forecast on identical
selection folds.

### Governed extraction from documents

An agent host may ask an LLM to extract numeric rows from tickets, schedules,
weather responses, planning documents, or other already-retrieved sources via
the `context_investigation` workflow. This is extraction, not generation:

- every row names its source document and carries one verbatim quote containing
  both the numeric value and its source-time token;
- the model normalises that token to an explicit-offset timestamp, but Gnomon
  verifies the normalisation and rejects uncited values;
- `known_at` is supplied per source document by the host when it acquired that
  source (with the run cutoff as an explicit fallback) and cannot be authored
  or backdated by the model;
- validated rows pass through the same inline-covariate loader and bitemporal
  snapshot as caller-authored data;
- the resulting feature still has no authority until identical-fold,
  point-in-time ablation admits it against the primary model.

The receipt labels rows `llm_extracted_host_verified`, retains their quotes,
and seals the complete table. Multiple independently extracted tables are
retained but not silently merged; the host must resolve their schemas and
provenance explicitly. Qualitative statements remain context claims or
prior-assisted candidates rather than being converted into invented numbers.

## TSFM capability matrix

`gnomon capabilities` reports `models.tsfm_capabilities`. These are adapter-level
capabilities verified through Gnomon's actual protocol, not upstream marketing
claims. It distinguishes multivariate targets from future-known covariates.

The current TSFM adapter protocol is univariate. Consequently, no TSFM is
advertised as covariate-aware yet, even when its upstream model supports such
inputs. The built-in `covariate_linear` candidate provides the first safe
covariate path. Future adapter implementations must add contract tests before
their capability flag can become true.

## Current limits

- Future-known numeric covariates only
- CSV or inline covariate rows; governed document extraction currently binds
  one validated table per forecast call
- Admission currently compares against a built-in selected model; if a TSFM
  wins the univariate evaluation, Gnomon reports that admission is unavailable
- No automatic web retrieval or arbitrary URL loading
