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

## New-development live experiment 004 — complete

192/192 decisions completed: eight new series, four registered origins, two
requested agent seeds, three arms. All selections were explicit; zero fallbacks
and zero harness failures. Mean per-case RMSLE:

| Arm | RMSLE | Reduction versus no ledger |
| --- | ---: | ---: |
| No ledger | 0.5396492493 | — |
| Original MAE ledger cards | 0.5417624406 | -0.3916% |
| Development RMSLE ledger cards | 0.5338399528 | 1.0765% |

Development cards improve 1.4624% over original cards. This does not establish
20%, statistical superiority or a final-test result. There were 396 API calls,
1,082,015 reported prompt tokens and 109,236 completion tokens. Engy did not
return monetary costs; those remain unknown. Full per-case outcomes and raw
hashes are in `evidence/new-agent-004.json`; raw transcripts remain in
`results/ledger-optimization/new-agent-004/`.

Independently recomputed every candidate score on this exact cohort from
predictions and actuals. The future-aware best-candidate bound is 0.4764364289,
only **11.7137%** below the observed control. Therefore no selection rule among
these eight fixed forecasts can achieve 20% on this development cohort. This
does not bound performance on unseen series or with new forecasts. The oracle
uses outcomes and is never presented to the agent or counted as an executable
policy. Reproduction: `headroom.py`; results:
`evidence/new-agent-004-headroom.json`.

## Exploratory forecast calibration — outside the registered objective

Retained `calibration_feasibility.py` and
`evidence/new-development-calibration-feasibility.json` as a separately labeled
feasibility screen. Both automatic arms can shift predictions in log space;
control estimates the shift from the latest two archived mature origins and
ledger variants can use more historical outcomes. Initial two origins stay raw
because earlier CV residual vectors were not archived. This archive-based
control is a prototype, not a finished live-agent or original-CV comparison.

Control RMSLE is 0.5587208311; the best full-development ledger window gives
0.5458925408 (about 2.3% lower). Variants were explored, so this is development
selection evidence only. No provider calls, source mutations or final-test
access occurred. Changing predictions falls outside the fixed-portfolio
protocol; do not silently count this result toward the registered objective.

## Decision point

The 20% target is **not achieved**. Repeating fixed-portfolio agent calls cannot
overcome the observed cohort's oracle bound. Asked the user whether the next
phase may expand to ensembles/calibration with identical capabilities in all
arms. No expansion or final evaluation is authorized by elapsed waiting time.
The 24-series confirmation partition remains reserved and unscored. Main and
PyPI remain unchanged; all implementation changes are on the development branch.

## Final-analysis preparation while scope decision is pending

Implemented `analysis.py` using the already registered 5,000-replicate,
seed-20260911 series/shared-circular-four-origin-block bootstrap. Added strict
cohort validation, seed/series/cold/mature breakdowns, retained fallback counts,
absolute and relative paired intervals, and explicit zero-control semantics.
The numerical gate cannot by itself mark the target achieved. The implementation
rejects our sparse four-origin agent runs for this consecutive-origin analysis;
it does not fabricate intervals by treating distant origins as adjacent.

Eleven synthetic analysis tests pass, covering exact paired effects,
determinism, invalid cohorts, zero controls and fallback inclusion. This work
uses no additional API calls and no confirmation data. The broader forecasting
scope decision remains pending; no user response is inferred from automatic
goal continuations or elapsed time.

## Scope clarification received

The user clarified that the desired edge is accumulated evidence of what has
and has not worked, through ledger infrastructure. The earlier question about
expanding forecast capabilities is resolved in favor of keeping the ledger
focus. Ensembles and bias correction are not the next phase. Updated PLAN.md
to prioritize contextual evidence capture/retrieval, comparable cohorts,
recency and evidence sufficiency, with unchanged forecasting candidates and
matched controls. The 11.7% oracle bound concerns the evaluated development
cohort, not a universal limit on ledger value. No claim of 20% improvement is
made; confirmation remains untouched.

## Contextual retrieval development started after user direction to continue

User explicitly asked to continue toward the target through ledger
infrastructure, with the same tools/information and improved robustness through
accumulated experience. The new CONTEXT_PROTOCOL.md makes the next control
stronger: every arm receives identical matured historical score rows and
forecast-time context, while the ledger arms organize that evidence. These runs
must not be pooled with earlier comparisons whose control had no historical
score rows. Original candidates and forecast/selection tools remain fixed.

Implemented a public `retrieve_context` operation: explicit progressively broader
filters, first cohort meeting a caller-specified matched-origin count, one SQLite
read snapshot, no provider selection/calls/writes. Do not choose cohorts by their
favorable observed losses. Nine focused tests cover selection/broadening,
recording visibility, invalid hierarchies, public dispatch and no writes/calls.

Preparation 005 initially failed on equivalent timestamp representations
(microseconds in normalized ledger output versus absent microseconds in cached
inputs). The failed copy remains in `results/ledger-optimization/context-memory-005/`;
its log is `/tmp/ledger-context-005-prepare.log`. No API calls or scored trial
result occurred. Fixed timestamp joins by comparing parsed instants. A synthetic
integration test also exposed an overly broad assumption that all providers
share one revision; preparation now reads each provider's recorded revision.
The ongoing StatsForecast preparation uses equal revisions and is unaffected
by that generalization. The source ledger is never mutated.

Full regression run: 1,193 passed, 29 skipped, one documented-workflow test
failed because its freshly read wall-clock cutoff was about 2.5 seconds earlier
than the preceding forecast recording timestamp. The package correctly rejected
that cutoff. The exact test passed on isolated retry without code changes.
Retained `/tmp/ledger-context-full-tests.log`; do not report the initial full run
as entirely green or attribute the clock anomaly to the new retrieval logic.

## Equal-information context trial 005 — complete and audited

All 192 decisions completed with explicit selection, no fallback and no harness
failure. The transcript audit verified identical shared raw history, current
information and system messages across all arms; selected fixed forecasts and
independently recomputed RMSLE match exactly (maximum numerical delta zero).

Mean RMSLE: raw-history control 0.5414427719; original MAE cards 0.5439300159;
contextual retrieval 0.5407181342. Context improves **0.1338%** against control
and does not establish the target. Worst-decile mean is 1.06094 versus 1.11867
for control, but provider-name agreement across seeds falls from 29/32 to 21/32
cases. Context wins eight pairs, loses nine and ties 47. These mixed development
diagnostics are not statistical proof of robustness. All cold-start cases remain
in the primary denominator. There were 410 API calls, 2,342,754 reported input
tokens and 117,363 output tokens; monetary cost remains unreported, not zero.

Reports: `context-agent-005.json`, `context-agent-005-audit.json` and
`context-agent-005-robustness.json`. Raw traces remain in the hash-linked run
directory. No confirmation data were accessed.

## Subsequent automatic screens

`cv_reliability.py` screened 24 past-only rules for the relationship between
previous CV and realized loss. The training-selected rule loses 0.8576% against
current CV on development validation. Retain the failure; do not promote it.
This does not modify provider forecasts. Any live pooled-history treatment would
require the same pooled CV/outcome information to be available to its control.

`support_screen.py` screened 108 descriptive support rules for changing current
CV's provider choice. The rule selected on origins 0..17 is all history, minimum
four matched origins, at least 50% paired wins, lower mean loss, no extra margin
or recent gate. It improves 1.9086% over automatic CV on origins 18..25. This is
development selection evidence only. Freeze this rule for live trial 008; keep
all three arms on the same raw historical data and unchanged forecasting tools.

## Retrieval parsing reuse and regression validation

A late-origin five-cohort profiling probe spent most time normalizing repeated
timestamps and decoding identical immutable executions. Added bounded per-call
timestamp reuse and per-read-snapshot execution reuse. Neither survives across
queries. The profiling probe fell from 2.523 to 0.890 seconds; this single probe
is not a general throughput benchmark. Every complete result across all 208
development queries equals its archived pre-optimization result, with zero
provider calls or ledger writes. A regression test confirms later actual
revisions remain visible on subsequent queries.

Full regression after the optimization: **1,203 passed, 29 skipped**. Logs:
`/tmp/ledger-context-final-tests.log`, `/tmp/ledger-context-parsing-equivalence.log`.
Hash-linked equivalence report: `evidence/context-parsing-equivalence.json`.

## Concise historical support trial 008 — complete and audited

192/192 explicit decisions, no fallbacks or harness failures. The audit again
verifies identical shared information/system messages, task-bound unchanged
forecasts, execution budgets and exact independently recalculated scores.
Mean RMSLE: raw-history control **0.5461489361**, original MAE cards
**0.5379198671**, historical-support packet **0.5271584056**. The treatment
improves **3.4772%** over control and about 2.0% over original cards.

Treatment wins 11 pairs, loses one and ties 52. Mature-history mean is
0.5078140619 versus control 0.5325603814 (about 4.65% lower). Worst-decile mean
is 1.0734213 versus 1.1436647. Seed agreement is 26/32 versus 29/32 provider-name
matches, so this does not establish universal robustness. Keep all cold starts
in the primary score. 398 API calls; 2,093,607 input and 131,896 output tokens
reported; no monetary prices returned. This is development evidence only.

The rule for selecting the confirmation implementation was written before
aggregate trial-008 scores were read: larger relative gain against each trial's
own matched equal-information control, exact ties broken by lower input tokens.
Thus choose `ledger_supported` (3.4772%) over `ledger_context` (0.1338%). No
confirmation observations or forecast scores have been inspected to make this
choice. Code and configuration will be frozen before confirmation preparation.

## Richer pooled memory selector prototype 009

Screened 16 error-estimation rules over past matured development episodes.
ExtraTrees predicts provider errors using observable history, current CV and
forecast totals; it never modifies or combines the fixed demand forecasts.
The training-selected rule improves only 0.7497% on development validation.
This complexity is not promoted. It would additionally require the control to
receive the same pooled historical information in any live trial. No API calls
or confirmation access occurred. Optional scientific dependencies remain confined
to the experiment environment, not Gnomon's runtime requirements.

## Confirmation guard preparation

Added a freeze binding code/source hashes, protocol, selected audited development
run, 24 series, 26 origins, two seeds, three arms and exact budgets. Confirmation
preparation requires this freeze and does not emit aggregate policy scores. The
agent runner rejects partial confirmation cohorts and mismatched prepared input
hashes. Eight synthetic guard tests passed without reading any held-out target
values. Development outcomes and final confirmation are not interchangeable.
