# Development journal

## 2026-09-11 — protocol and feasibility

Protocol committed as `91ff5d7` before optimization experiments. Target remains
20% versus the same no-ledger agent, with the stipulated final evaluation gate.

Located the legacy Favorita 1.1.9 run; froze its first 85 completed origins,
340 cases on four series. These are development data. The source corpus hash
and 85 individual origin hashes are retained in `evidence/legacy-85-screen.json`.
Frozen extracted inputs: `results/ledger-optimization/legacy-85-frozen-v2/`.
Original files and SQLite ledger remain untouched.

Recomputed the recorded arm scores independently. Legacy mean case RMSLE:
no-ledger 0.5450533844; ledger 0.5385710449; direct StatsForecast agent 0.5416446984.
Hindsight best-candidate oracle: 0.4755035724. Thus the maximum possible gain
over this legacy no-ledger baseline is 12.7602%, even with future knowledge.
This oracle is a feasibility ceiling, not a deployable selector.

The first extraction attempt stopped because legacy fallback records can have
`prediction: null`; corrected extraction uses their explicitly recorded
`scored_provider` prediction only when `used_fallback` is true. No original
scores were replaced. All reconstructed arm scores agree within 1e-10.

## Development policy screen

Nineteen preregistered simple policies (current CV plus MAE/RMSLE history,
4/12/all-origin windows, fixed CV weights) use only already matured same-series
outcomes. API calls: zero. Best full-development score among the screened
variants is 0.5272016079 (RMSLE lifetime/CV equal blend), 3.2752% below the old
control. This is selected development performance, not an unbiased effect.

The early development slice prefers lifetime RMSLE alone; the later slice
prefers a 12-origin/CV blend. Preserve all variant scores and case selections;
do not present the best retrospective slice as a preregistered winner.

## Product change 001

`compare_history` and `compare_context` gain optional `metric`, `recent_origins`
and `negative_predictions`. Existing MAE defaults and immutable evidence stay
intact. Calculated ranks, exact ties, differences, explicit denominators and
recent/lifetime disagreement are linked to the full matched-origin evidence.
Negative actuals invalidate RMSLE; prediction clipping is explicit. Full suite
at the initial candidate: 1,166 passed, 29 skipped. Four runner tests also pass.

## Live Engy development pilot

Three matched arms at origin 12 for all four legacy series, requested seed 7.
Model: deepseek-v4-flash-0731. Max 3 forecast attempts / 6 chat turns / 2048
output tokens per turn. All arms use the same typed execution selection helper.
Provider execution is task-validated playback of the saved StatsForecast
forecasts, not newly fitted models. No case actual or hindsight score enters
the agent prompt. The source ledger enforces origin-time visibility.

Pilot 001 exposed a harness defect: the playback adapter omitted covariate
capabilities and Gnomon rejected those requests. The attempt ended with a
harness exception; it is excluded from accuracy conclusions, not scored as a
ledger failure. Its per-call usage was not durably saved before the exception,
so its cost/call count is unknown, not zero. Retained log:
`results/ledger-optimization/agent-pilot-001/harness.log`.

Pilot 002 declares matching covariate capabilities, catches structured tool
rejections, and journals each API response immediately. Its manifest and raw
responses are under `results/ledger-optimization/agent-pilot-002/`. Engy reports
token usage but no price in the responses observed so far; dollar cost remains
unknown. No fixed zero-dollar claim is made.

## Outstanding final-evaluation prerequisite

The local legacy results are available, but the original Favorita runner and
full corpus have not been located. Requested the current pod SSH address or
project/data path from the user while continuing useful local work. No final
holdout has been opened, selected by outcomes, or scored.
