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

## Expanded experiment 003 — complete

128/128 decisions completed, 32 matched series/origin/seed pairs, no fallbacks.
Mean RMSLE: no ledger 0.5537699094; old MAE cards 0.5454432126; RMSLE cards
0.5449298781; explicit CV/history blend suggestion 0.5585197485. The RMSLE
cards improve 1.5963% against control and only 0.0941% against the old cards.
The blend suggestion is 0.8577% worse than control and is not promoted. A useful
automatic offline selector did not translate into a useful agent suggestion.
Retain this failure. 267 API calls, token usage retained, dollar costs unknown.
One malformed tool request was repaired; all final selections were explicit.

## Data-access prerequisite resolved

Recovered the Arena branch using the retained deployment record and fetched
commit `b600eaa2c2691bebed926dac996d4b03e0c216e9` into
`/tmp/gnomon-ledger-opt-arena`. Read-only SSH to the original Arena pod also
works. The raw data was not present there, so downloaded the same raw Favorita
archive through `datasetsforecast` locally. No source run or remote service was
changed. User does not need to provide an SSH address now.

`PANEL_PROTOCOL.md` was committed as `97d8c55` before new panel selection. It
reserves disjoint items/stores, eight additional development series and 24
confirmation series. Future confirmation values are moved only by deterministic
data preparation and must not be analyzed until the final candidate is frozen.

## New panel preparation and additional development

All 32 disjoint pairs qualified under the registered rule. The ID/file/source
hash manifest is retained in `evidence/new-panel-manifest.json`. Confirmation
has not been forecast, scored or used for tuning. Original-source assumptions
(zero-fill absent sales, clip returns, known future promotions) are disclosed.

Generated all eight fixed recipes and two historical CV folds at 26 origins
for the eight new development series: 208 cases. The current-CV automatic
selector scores 0.5658325868 RMSLE; hindsight best-candidate oracle is
0.4942338380 (12.6537% headroom versus CV). This is not yet a live agent
control. Simple past-only selectors improve only modestly (best full-development
score 0.5566543426). A 24-variant calibration/retrieval screen selects pooled
16-neighbor retrieval on origins 0–17; its development validation RMSLE on
18–25 is 0.5479491984. No final-test claim follows from these screens.

Preparation's initial readback included the current pending shadow origin in
legacy card exclusion counts, whereas the original Arena query ends at the
previous origin. Corrected that query window, retained the initial inputs,
and generated `new-development-input-v2` with unchanged forecasts/metrics.
The upcoming live cases' 72 historical card windows are checked directly
against the real ledger, including models, matched counts and exclusions.
An initial verification compared different provider display orders; matching
the original provider order resolves that verification mismatch without changing
any scores.

New-development live protocol: 64 matched case/seed pairs, 192 decisions;
no ledger vs original MAE cards vs development RMSLE cards. Forecaster families,
completion rules and budgets remain identical. Additional model calibration
ideas remain prototypes and are not silently substituted into this comparison.
