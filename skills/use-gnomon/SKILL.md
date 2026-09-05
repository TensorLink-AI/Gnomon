---
name: use-gnomon
description: Use Gnomon's available tools to inspect time series, compute exact statistics, forecast with registered providers, and evaluate explicit historical evidence.
---

# Use Gnomon

Use the schemas actually exposed by the current session. The default execution
profile and explicit legacy profiles have different contracts; do not mix arguments.
Discover capabilities when provider names, supported inputs or storage are unknown.

## Default execution session

- Inspect file/store data with `gnomon_inspect`, specifying known column names.
  Reuse its frozen `data_ref`; for panels select a returned series explicitly.
  Disclose knowledge-time assumptions, declared units and repairs.
- For an observed quantity, call `gnomon_describe` with the exact statistic:
  mean, median, latest, minimum, maximum or sum. Optional start/end timestamps
  select an inclusive window. Do not substitute latest for average.
- For a forecast, name an explicit registered `provider` and supply either a
  typed `request` or a `data_ref` plus positive `horizon`. Respect the user's
  preferred model; never substitute a reference baseline without disclosure.
  If no provider is specified, clarify when the choice matters or explicitly
  identify a baseline result as a reference forecast.
- Map an explicit duration onto the disclosed grid (daily next week: seven steps).
  Missing frequency/timezone semantics require clarification or explicit assumptions,
  not invented dates. Known covariates must obey request capability and time cutoffs.
- Provider URLs, imports, authentication and ledger paths are operator startup
  configuration, not tool arguments. Never put secrets in prompts.

Inference alone does not prove accuracy, calibrated uncertainty or action authority.
Preserve provider/revision identity, uncertainty, snapshot and execution identifiers;
unknown model weights or training cutoffs stay unknown. A rejected capability is not
permission to drop an input silently.

## Evaluation and retained evidence

Use `gnomon_evaluate` when comparison is needed: name candidates, an explicit
baseline, horizon and suitable fold/budget settings. It can incur provider calls;
respect task scope and operator ceilings. Its compact study ID retrieves full evidence.
Count incomplete folds and failures, not only successful forecasts.

If a response has `partial: true` and `result_ref`, use `gnomon_read`; do not treat
the scalar summary as the full result. A JSON pointer selects an exact needed field.
Otherwise concatenate returned `text` pages at `next_offset` until null before
parsing JSON. References expire with session closure or retention eviction.
`RESULT_RETENTION_LIMIT` can mean execution already completed; inspect its receipt
and durable ledger identifiers before considering a retry that may repeat work.

With a configured ledger, `gnomon_route` requires one original study, matching
providers and explicit source-availability and local-recording cutoffs. An explicit
baseline fallback means the evidence was insufficient, not that the baseline won.
Rescoring appends a new study without changing original predictions.

Use `gnomon_ledger` only when exposed. Queries and score creation differ from
outcome/import writes, which require operator authorization and user task scope.
Keep valid time, source availability and local recording time distinct.

## Optional explicit temporal calculations

When `gnomon_temporal` is exposed, use it for date shifts, instant normalization,
elapsed duration, half-open interval relations or event ordering. Do not invent an
implicit current time. Calendar days and elapsed 24-hour periods differ at clock
changes; select the intended mode. Local timestamps require a named timezone;
ambiguous times require an explicit fold, and nonexistent times are rejected.
Dates are not midnight instants. Tied event timestamps do not establish causality
or source availability. This tool calculates supplied facts; it does not verify them.

## Advanced legacy sessions

Only when an explicit core/evidence/decision/data/full profile exposes the older
evaluated/context tools, read [advanced guidance](references/legacy-workflows.md).
Do not send legacy `candidates`, `threshold`, `context_events` or publication
arguments to the default forecast tool. Unsupported trend/anomaly/decision operations
need an appropriate exposed tool, not an invented extension to exact describe.

## Feedback

Record feedback only after explicit agreement. Use `gnomon-feedback create` for
minimal metadata; private notes stay local. Preview the exact shareable receipt
with `gnomon-feedback preview` before asking for separate export/submission consent.
Never add `--consent` on the user's behalf. Do not record raw series, prompts,
credentials or full arguments. Do not reward call volume.
