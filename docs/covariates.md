# Covariate enrichment

Aion can evaluate externally sourced, future-known variables such as published
holiday calendars, scheduled prices, or archived weather-forecast vintages.
The agent decides what might matter and fetches it. Aion owns the temporal data
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
historical fold cutoff, Aion selects the most recent row whose `known_at` is no
later than the cutoff.

Do not label a reconstructed historical observation as a forecast. For example,
today's archive of observed temperatures does not reveal what weather forecast
was available six months ago. Weather inputs require archived issued forecasts.
If those vintages are unavailable, Aion rejects the proposal for backtesting.

This release accepts numeric `continuous` and `binary` features whose
availability is explicitly `future_known`. It does not silently encode
categoricals, interpolate gaps, or fetch URLs.

## CLI workflow

Get the exact forecast timestamps and selection-fold cutoffs:

```bash
aion covariates guide sales.csv \
  --time timestamp --target sales --frequency D --horizon 14
```

Validate a locally fetched proposal:

```bash
aion covariates validate sales.csv \
  --time timestamp --target sales --frequency D --horizon 14 \
  --covariates covariates.csv \
  --covariate-mapping 'is_holiday:binary:future_known,temperature_forecast:continuous:future_known'
```

Evaluate it and produce a forecast:

```bash
aion forecast sales.csv \
  --time timestamp --target sales --frequency D --horizon 14 \
  --covariates covariates.csv \
  --covariate-mapping 'is_holiday:binary:future_known,temperature_forecast:continuous:future_known'
```

For panel data, pass both `--series` for the target file and
`--covariate-series` for the covariate file.

## Admission procedure

Aion always retains the evaluated univariate forecast as the control. Candidate
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

## Agent and MCP workflow

When MCP is enabled, agents can call `aion_covariate_guide`,
`aion_validate_covariates`, and `aion_propose_covariates`.

The tools accept local files. Agents should use their own permitted web tools to
retrieve data and preserve its source URL separately; Aion will not dereference
arbitrary URLs. `aion_propose_covariates` runs the same admission gate as the CLI.

## TSFM capability matrix

`aion capabilities` reports `models.tsfm_capabilities`. These are adapter-level
capabilities verified through Aion's actual protocol, not upstream marketing
claims. It distinguishes multivariate targets from future-known covariates.

The current TSFM adapter protocol is univariate. Consequently, no TSFM is
advertised as covariate-aware yet, even when its upstream model supports such
inputs. The built-in `covariate_linear` candidate provides the first safe
covariate path. Future adapter implementations must add contract tests before
their capability flag can become true.

## Combining covariates with context events

A run may pass both `--covariates` and `--context`. Each enrichment is still
ablated on its own, and then an adjudication stage compares every candidate —
history-only, plus context, plus covariates, plus both — on identical folds:
the same rolling origins, the same observation prefix at each origin, and
enrichment inputs gated by `known_at` at that origin's cutoff.

The winner is the one that earns its complexity. Each rung — one enrichment,
then two — must beat the standing champion by
`minimum_baseline_improvement`, so:

- enrichments that only clear the bar *together* are compared against the
  history-only forecast, and can be admitted;
- a second enrichment that merely re-encodes the first is compared against
  the winning single, and cannot be admitted on a rounding-error gain.

The candidate sets are exactly what the ablations produced: every
backtest-admissible event, and the covariates the ablation retained. When the
ablation retained none, the full declared set enters the ladder instead — a
covariate that cannot carry a forecast alone may still carry one alongside
context, and the margin rule is what keeps that honest.

A joint winner reports `selected_model: covariate_linear+event_adjusted` —
the covariate model's forecast carrying the part of the event effect that
model did not already explain. Every adjudicated series records an
`enrichment_adjudication` evidence record with the shared folds, each
candidate's per-fold scores, and the comparison that decided the winner, so
the choice can be audited rather than taken on trust.

With a single enrichment type nothing changes: one ablation already *is* the
two-candidate comparison, and no ladder convenes.

## Current limits

- Future-known numeric covariates only
- CSV covariate files
- Adjudication requires at least four rolling folds and a built-in
  univariate comparison model; when a TSFM, ensemble, or VAR forecast wins
  the history-only evaluation, the ladder records why it stood down
- Admission currently compares against a built-in selected model; if a TSFM
  wins the univariate evaluation, Aion reports that admission is unavailable
- No automatic web retrieval or arbitrary URL loading
