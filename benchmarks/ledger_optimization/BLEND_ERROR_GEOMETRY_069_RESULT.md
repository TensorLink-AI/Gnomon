# Diagnostic069: forecast range explains only part of remaining blend error

Freeze053d1d7 preceded calculation. All416 original-development cases retained;
no forecasts, weight fits, API calls, policy changes or protected-data access.
Four synthetic tests passed. Independent33,816-check audit passed for every
production/CV projection, per-point identity, counts and aggregates.

| Quantity | Current-only blend | Ledger blend |
|---|---:|---:|
| Actual mean case RMSLE |0.2576259533432357|0.25107820237988004|
| Hindsight per-step range-floor mean RMSLE |0.10426052019657674|0.10441045171335232|
| Current-CV range-floor mean RMSLE |0.11763734790387013|0.11775990491405805|
| Actual below all forecasts, points |1797|1805|
| Actual inside forecast range, points |6623|6608|
| Actual above all forecasts, points |1564|1571|
| Total scored forecast points |9984|9984|

For ledger,66.19%of future points are inside the seven-forecast range. The
projection floor contributes23.5603%of pooled squared log error; deviation of
the chosen blend from the projection contributes52.5869%, and the nonnegative
cross term contributes23.8528%. They sum to100%, but are not fractions of mean
case RMSLE or causal explanations. The cross term cannot be assigned entirely
to forecast coverage or blend quality. Per-domain floor energy shares are20.03%
electricity and23.91%pedestrian. Full vectors and denominators are retained.

The projected future floor (.10441) is lower, not higher, than the current-CV
floor (.11776) overall. This does not support the specific explanation that
future outcomes were universally farther outside the models' range than the
backtest outcomes. It does not rule out shifts in specific tasks, or establish
what information would make the right blend predictable. Mean signed log error
.01161also does not by itself justify a causal bias correction.

Interpretation: forecast-range limitations are real but do not account for all
remaining error. A diagnostic projection uses current future actuals and permits
independent weights at every forecast step. It is outside068's four-block action;
its .10441score is neither a deployable result nor the optimum of068's action.
Do not pass projected values/weights into future learners or agents.

Next bounded hypothesis: test a shared finer-time blending action, using the
same predecision/matured evidence and computation accounting in both arms. Keep
matched control, original strong block-CV and061incumbent guards. A larger
hypothesis class may overfit current CV, so retain failure and require the same
20%target; do not accept a weak control as apparent ledger progress. Freeze the
learner and numerical certificate before scoring. No paid/final confirmation
is justified by a hindsight diagnostic.

Run1.406s, zero new fits/API; inherited49,616raw forecasts and832068weight fits
remain in their receipts. Evidence: results/blend-error-geometry-069-001 and
its tracked receipt/archive. Main/PyPI unchanged. Goal remains unproven; current
best original-development prospective-rule score is still068's .2510782, not the
projection floor. No new Hermes trial or held-out accuracy result.
