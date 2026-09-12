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

## Frozen confirmation 010 — complete; objective failed

Freeze 1709b4c was committed/pushed before preparing the 24 reserved series.
All 26 origins, two requested seeds and three equal-information arms completed:
3,744 decisions, 1,248 matched case/seed pairs. No optional stopping, partial
performance analysis or treatment changes occurred. The source/protocol hash
audit passed at completion.

Mean RMSLE: control **0.5272968020**, original ledger **0.5284976078**, support
**0.5243270335**. Support improves **0.5632%** versus control, with paired 95%
interval **−0.3620% to +1.8225%**. Versus original ledger, improvement is
0.7891% (interval −0.0596% to +2.0030%). Neither establishes reliable superiority.
The 3.48% development improvement did not generalize at that magnitude.

The independent future-aware candidate minimum is **0.4614605508**, while the
20% target requires **0.4218374416**. Thus the maximum possible selection-only
improvement on this exact cohort/control is **12.4856%**. Repeating selection
tuning cannot meet the registered 20% objective here. Do not weaken the control,
change the score, cherry-pick origins or present reused confirmation cases as
untouched. Further target pursuit requires an explicitly revised test/objective
scope; none was silently substituted. This does not bound all unseen datasets.

All decisions ended in explicit execution selections. Zero API errors, harness
failures and final fallbacks. Eleven intermediate tool calls were rejected and
recovered: five over-budget forecast requests, five premature/conflicting
selections, one invalid forecast request. All actual forecast attempts remained
within the same limit. Prepared StatsForecast candidates/CV disclosed no recipe
fallbacks. Execution reliability is separate from a forecasting improvement.

The transcript audit verifies identical shared information, unchanged candidate
points and independently reproduced RMSLE for every decision. A second read-only
audit checks 4,992 executions, 7,800 historical-origin exposures, 109,200 actual
visibility checks and 124,800 raw score checks, preserving the ledger hash.
Support wins 146 pairs, loses 105 and ties 997. Seed provider-name agreement
falls from control 571/624 to 509/624; tail improvement is very small. A broad
robustness claim is therefore unsupported.

7,804 API calls; 41,106,205 input and 2,350,618 output tokens reported. Monetary
cost was not returned. Raw evidence remains locally hash-linked; all summaries,
confidence intervals, audits and the negative objective result are retained in
`evidence/confirmation-agent-010*`. The reviewed record explains inherited
"development" prose in immutable frozen tool outputs whose actual scope is
confirmation. Original artifacts and numeric results were not rewritten.

Post-run audit/plot scripts in `audits/` were added after the experiment and
do not change the frozen treatment. See `CONFIRMATION_010.md`. Main and PyPI
were not modified for this work. The goal is not marked complete.

## 2026-09-12: CV-leader context screen 011, not promoted

Revalidated the active 20% forecast-error objective and the original negative
confirmation. The separate workflow-efficiency experiment and Hermes skill
comparison do not satisfy the three-arm, untouched forecast-error objective.
The original confirmation cases/ledger artifacts remain available locally;
missing `/tmp` Hermes archives were not recreated or claimed as present.

Committed protocol, code and five passing tests as bd9768f before running a
bounded screen on only the existing 208 development cases. Four fixed variants
conditioned retrieval on the same historical CV leader, requiring 4/8 origins
and 50%/75% paired wins. Same providers and immutable predictions, no API calls.
Training selected minimum 4 / 75% wins; the later development slice was 0.28%
worse than CV and 2.23% worse than existing unconditioned support. Do not promote.

Independent metric calculations agreed with every cached candidate loss. All
208 reference-policy choices agreed with screen 007, input hash unchanged.
Future outcomes, late recording and source visibility were covered by tests.
Retained every variant and per-case retrieval/choice/loss in cv-context-011.json;
no after-the-fact variant substitution or fresh confirmation was performed.
No forecasting superiority claim, main merge, release or PyPI change.

## 2026-09-12: override-outcome trust screen 012, not promoted

Committed protocol, implementation and tests as b3bfe38 before scoring. Four
fixed gates require 2/4 matured earlier override decisions, using all history or
the latest four. Reconstruct historical support proposals using only evidence
visible at their own origins. Permit a new override only if past overrides beat
their contemporaneous CV choices in mean loss and on at least half the episodes.
No changed predictions, API/provider calls or new confirmation observations.

Training selected minimum two / latest four. Later development RMSLE is
0.5426050080 versus CV 0.5456698185 and existing support 0.5352551237:
0.56% better than CV, 1.37% worse than incumbent. Seventeen incumbent overrides
were rejected, producing seven improvements, nine regressions and one tie.
Do not promote or replace the selected variant after inspecting later scores.

Ten focused tests passed (five new plus five shared). All 208 incumbent choices
match screen 007, input hash remains unchanged, and retained report bytes match
the original local output. Every cached candidate loss was recomputed. Preserve
all gates and per-case evidence in evidence/override-trust-012.json. The protocol
and result explicitly retain inherited replay assumptions and the already-used
development scope. The original target remains unestablished; main/PyPI and the
spent confirmation are unchanged.

## 2026-09-12: opportunity and historical-linkage audit 013

Previous goal turn made progress by retaining the negative screen-012 result
and pushing aec5e89. Revalidated branch state and the unchanged objective before
this audit. No live process or pending paid run was inferred from old files.

Registered 45dc81b before computing a fixed 256-trial historical provider-label
corruption diagnostic and future-aware headroom on development only. Correct
support beats every corrupted-history aggregate overall and later, while cold
origins are unchanged. This is diagnostic sensitivity to truthful linkage, not
a fair no-ledger comparator, p-value or uncertainty interval.

All-case RMSLE: CV 0.5658325868, support 0.5540610449, future-aware per-case oracle
0.4942338380. Maximum gain is 12.65% against CV; later slice maximum is 9.14%.
Correct support captures 16.44% of the all-case oracle opportunity. No basis to
keep buying selection-only trials on this panel in pursuit of 20%, and no
permission to select a future holdout by favorable oracle headroom.

Ten tests passed. Independent post-run recomputation checked 53,248 mapped
selections, 1,024 aggregates and 208 prior incumbent choices; input and frozen
source hashes unchanged. Preserve all mappings and choices, including identity
mappings, in evidence/opportunity-013.json. No API/provider calls or new data
access. FORECAST_EVALUATION_GATE.md now records the development-only feasibility,
fair-agent comparison and immutable final-test gates. The original target and
negative confirmation remain unchanged; no main or PyPI changes.

## 2026-09-12: M5 prospective source preparation 014

Previous turn made progress with audit 013 and the fair-evaluation gate, pushed
as 50a60c4. Revalidated state and inspected the separate FreshRetailNet ROI plan
without modifying it. Its 90-day source cannot meet the existing annual-history
and 26 nonoverlapping 14-day origin schedule. Selected the longer M5 retail
source for task relevance before computing any candidate outcomes, not because
of an observed advantage. No M5 references were found in the searched prior
local experiment manifests/receipts; this is not a global contamination proof.

Committed source pin, protocol, prefix-only selection and tests as ce5c03c before
download. First preparation failed on missing calendar d field before target
export. Preserved the failure, corrected the mirror's one-based date mapping
under 0cce4b4, then retried without changing sampling rules. All seven tests pass.

Eight development series (CA_3/WI_2), 24 reserved series (eight other stores), all
32 items unique. 30,490 prefixes checked; 27,380 eligible, 3,110 below initial
nonzero-history threshold. Development exports 5,840 rows and supports 208
origin/horizon pairs. Two clean preparations agree byte-for-byte on development
targets and exactly on split/eligibility metadata. All source and code hashes,
timestamps, disjointness and planned horizons checked. Reserved later targets
were not inspected numerically, scored or exported; no forecasting model or
Engy call ran. Keep source/recording assumptions and unavailable promotion
feature explicit. Source amendment does not replace the primary target or grant
permission to cherry-pick reserved outcomes. Full commands, failure, retry,
timings and receipts are retained in evidence/m5-*-014*. Main/PyPI unchanged.

## 2026-09-12: M5 forecasting and fixed-policy screen 015

Previous turn made progress by freezing and preparing the new source under
prefix-only selection. Current turn froze 53ea3d9 before fitting. Used the same
eight pinned recipes and original numerical package versions in an isolated
runtime. A read-only uv cache error was resolved with a separate temporary cache
before model execution; no numerical version or forecasting rule was changed.

All 208 development cases completed, 1,664 production plus 3,328 CV requests,
1,792 exact-request cache misses and 3,200 hits. No numerical recipe fallback.
Support RMSLE 0.6257278285 versus CV 0.6169646542: 1.42% worse. Future-aware
minimum 0.5450017262 permits only 11.66% improvement against CV. Later slice
support is 2.24% worse; lifetime MAE's better later score is retained as a fixed
diagnostic, not selected retroactively or equated to a 1.1.9 agent.

Twelve pre-dispatch tests passed, plus one later independent-audit fault test.
The full independent audit reproduces 4,992 metrics/requests, 1,792 distinct
request identities, CV maturity, causal support selections and input/code hashes.
All 101 changed support choices retained: 55 improvements, 46 regressions,
negative net mean result. No case or duplicate prediction deleted. Measured
subprocess elapsed 170.78 seconds, no Engy calls. Original requests, predictions,
CV evidence, command logs and source snapshots remain locally hash-linked.

M5 failed the preregistered development gate. Its reserved stores stay closed;
no paid live selection trial is justified by this screen. Asked the user to
resolve the fixed-forecast scope: retain the negative conclusion, or explicitly
permit equal additional forecasting choices in all arms while retaining 20%.
No expansion occurred while that question was pending. The objective remains
unachieved; this is the first scope-blocking decision point after completing
the new source's development work. Main/PyPI and earlier evidence unchanged.


## 2026-09-13 (Brisbane): ML workflow audit and prospective maturation v3

The user subsequently authorized a common actual-model-fitting/backtesting task
for all arms. That is a scope amendment from the earlier fixed-eight-recipe
screen, not a reinterpretation of its negative results. Keep the 20% held-out
objective, matched-information rules and current-1.1.9 comparison requirement.
The Hermes/Gnomon 1.2.0 ML trials use development Favorita series; they do not
satisfy either final requirement.

The completed checkpoint-v1 development trial finished 96 sessions with 3,792
independent checks and no integrity failures. Full workflows: plain 29/32,
Gnomon 28/32, ledger 29/32. Mean per-case RMSLE including declared fallbacks:
plain 0.5064452434610625, Gnomon 0.5532628499601663,
ledger 0.5407122678674124. Ledger is 2.27% better than Gnomon and 6.77% worse
than plain. On the selected 22-case all-three-complete subset, ledger is 1.84%
worse than Gnomon; this subset is diagnostic, not an unbiased treatment estimate.
Raw archive SHA256: 476f33ed8092626ef025bf98c4cd43e612a69502fccbc00a4d8b4e27141fdc38.

Checkpoint-v2 introduced equal protected selection budgets and bounded corrective
continuations, tested in 26 regression groups including real pinned Hermes with
synthetic upstream responses. Its local pilot was interrupted by an environment
reset after 10/36 sessions, all ten full. The managed process was unavailable
and pinned /tmp runtimes disappeared. Preserve 159 forwarded requests, 157 saved
responses, and 2,472,360 reported tokens (147 responses had token usage); incomplete
receipts and unknown billing are explicit. Do not count the interruption as agent
failure or pool those ten observations with a restart. No promotion gate ran.

A read-only audit of v1 found 54 already-executed production forecasts at 28
matured origins, but only the 28 selected submissions received outcome scores.
The 26 unselected alternatives were omitted; a read-only SQLite check confirmed
all score-presence results and preserved all database hashes. None of 35 ledger
reviews displayed matched production comparisons. This is a benchmark integration
omission, not a demonstrated failure of TemporalLedger.evaluate. Compact audit
artifacts are evidence/ml-review-016*.json; originals remain unchanged.

Prospective checkpoint-v3 scores every eligible prior production execution once,
using exact origin, targets, series and unit identity and explicit outcome
visibility. All three arms get the same raw matured outcomes; ledger additionally
persists them through its public API. No new provider calls or revised submissions.
Keep the v2 model/prompt-budget rules and existing review ordering. The isolated
Targon CPU pod is reachable; restore frozen dependencies there and pass the full
preflight before any new Engy requests. Host migration is disclosed and results
will not be pooled across versions. Full protocol: ../hermes_ml_checkpoint_v3/PROTOCOL.md.

The previous status-only turn is classified as no progress toward performance;
this turn revalidated terminal local state and advances code, visibility tests,
audit closure checks and durable runtime preparation. The target remains unproven.
No main/PyPI changes, no reserved final outcomes opened.


V3 startup update: all 27 preflight groups passed, and an independent read of
six matured production executions reproduced MAE/RMSE/bias/RMSLE and confirmed
zero additional fits. The first paid-pilot startup then stopped at runtime parity,
before any API call. Pinned Hermes' optional Bedrock adapter had lazily installed
pip, boto3, botocore, s3transfer and jmespath during its synthetic plain-arm test.
The failing startup and package differences are preserved. Restore the original
inventories; set the existing HERMES_DISABLE_LAZY_INSTALLS=1 in every agent's
environment and add before/after inventory checks to preflight and inference.
Rerun the full preflight under this explicitly frozen environment. No forecasting
model, budget, task, score or paid outcome changed. This is an environment repair,
not a new candidate recipe or an accuracy-driven retry.


Read-only portfolio headroom audit 018 retained all 96 v1 cases and recomputed
all submitted errors. A hindsight minimum over each arm's already-executed
production alternatives plus its original graded forecast would reduce ledger
mean RMSLE from 0.54071227 to 0.52314926 (3.25%, three improved cases). Plain's
corresponding reduction is 1.37%, Gnomon's 4.57%. This restricted oracle is not a
bound over unexecuted configurations or a causal policy, and was not fed to the
agents. It indicates that 20% would require better configuration exploration,
not merely recovery/selection among the production predictions already present.
No new provider/ledger writes, changed original scores, or reserved data access.
The fresh v3 promotion rule remains completion/integrity only.


## 2026-09-13 (Brisbane): baseline compatibility and live audit 019

Previous goal turn made progress: committed/pushed the common all-execution
maturation fix, repaired a guarded zero-call startup, passed 28 regression groups
and launched a separate durable pod pilot. This turn verified the same live PID
and boot/start identity; no trial restart or source change. Copied evidence off
pod. The first six closed sessions all completed with zero API errors; the frozen
independent auditor reproduced 303 checks with zero failures. A partial progress
snapshot cannot pass the 36-session promotion gate or establish an accuracy gain.

Installed the published 1.1.9 wheel into an isolated local runtime; its build is
exactly the objective's 59a6d81709a4625bf042e7ca152aa5f12534c28a. Ran the same
synthetic public-API script against it and pinned 1.2.0. Both support request
identity/covariates, controlled-clock execution, recording/source cutoff exclusion,
pending/partial/strict/complete scoring, atomic batch scoring, score reuse, saved
retrieval and matched history. Independent metric expectations agree and scoring/
comparison made zero additional provider calls. First assertions expected list
rather than tuple points; corrected the tester for both versions and retained
both failed attempts/new databases. No package implementation was changed.
Pinned wheel and exact argv/stdout/stderr are retained under ignored
results/ledger-119-compatibility-019; compact receipt is evidence/ml-baseline-compatibility-019.json.

This closes a compatibility prerequisite, not the incumbent-performance gap.
PLAN.md now explicitly records the user's later common-model-fitting amendment
and requires prospective evidence-adapter semantics for a real matched 1.1.9
comparison. It forbids passing off the current 1.2.0 workflow or a deterministic
MAE selector as the incumbent agent. Reserved outcomes remain unopened and no
final candidate is eligible for dispatch. Main/PyPI unchanged.

Added a separate progress auditor that verifies frozen source hashes, reads only
closed immutable session snapshots, preserves every source-file hash, and writes
into a new output directory. An altered-auditor fixture was rejected before
import. Rechecking the six-case snapshot preserved source bytes and all checks.
No live result, model request, scoring rule, prompt or budget was changed.


A later closed snapshot includes 10/36 pilot sessions: plain 3/3, Gnomon 3/3,
ledger 4/4 full, zero completed-session HTTP errors. All 519 independent checks
pass. Eight unique prior production forecasts have matured, including four
unselected alternatives; all match the original targets and start zero new fits.
The partial run remains ineligible for promotion or a comparative accuracy claim.
Receipt: evidence/ml-v3-progress-020.json. The running source stays frozen at
931317a; these baseline/audit utilities are outside that experiment's code path.


## 2026-09-13 (Brisbane): production comparison fidelity audit 021

Previous goal turn made progress on the 1.1.9 compatibility prerequisite and
closed-session audits. This turn revalidated the same remote PID/boot/start
identity and mirrored new evidence without restarting or changing the trial.
The 14-session closed snapshot passed 767 frozen-auditor checks. A separate
read-only audit verifies all four distinct saved ledger review packets in that
snapshot contain a production comparison. Recomputed each displayed mean RMSLE
from task-matching original predictions and mature actuals, checked origins,
source/recording visibility and exact sample counts; one packet now has two
matched production origins. Cumulative copies of the same packet are counted
once. No original data, score, provider execution or ledger changed.

Six independent fixture tests pass, including wrong displayed values, wrong
origins, future recording, unmatured targets and duplicate matched origins. The
receipt is evidence/ml-review-coverage-021.json. This establishes that the fixed
maturation path produces correct displayed evidence. It is not proof that an
agent uses it well, a measured accuracy improvement or satisfaction of the20%
held-out objective. The final outcome and 1.1.9 agent comparison remain pending.

## 2026-09-13 (Brisbane): continuous development preparation 022

Previous goal turn made progress: revalidated the live remote process and audited
26 closed pilot sessions (all full), with 1,407 checks and zero integrity failures.
This turn retained that trial's frozen code and gate. Later closed evidence has
two no-execution service failures, one plain and one ledger: each received three
429 capacity-exhausted responses. Other sessions encountered 504 upstream miner
timeouts. Keep these cases and original fallbacks in the trial; do not attribute
their missing checkpoints to agent numerical-budget exhaustion or silently rerun.
The 30-session snapshot passes 1,578 integrity checks. Remote PID and boot/start
identity remained verified live; this partial snapshot is not a final gate.

Found a design limitation in the development schedule: rounds 0,1,2,3,22,23,24,25
provide at most seven earlier decisions, even in the calendar-late phase. This
cannot test the ten-origin accumulation hypothesis. Before exporting anything,
froze ML_CONTINUOUS_022.md and a validated preparation tool at a446c64. Prepare
all 26 consecutive origins on the same four already-used development series;
no new item/date selection based on error or hindsight, and no model execution.

First preparation rejected the source's integer history versus float actual
representation (28 versus 28.0). Retained its exact failed output. A diagnostic
found those overlapping values numerically identical. Frozen correction db57a4a
allows exact cross-role numeric equality and preserves each original role's
representation; within-role conflicts, changed values and booleans still reject.
All ten synthetic tests pass, including temporal perturbation, metadata exclusion,
source/output independence, overlap rejection and representation fidelity.

Two separate exports now agree byte-for-byte: 104 tasks, hash
cf7bdd21e216e84809edb8653710e0c0755402864201400c5749a8dbff00f561.
An independent verifier uses positional concatenation of only first/last anchor
arrays, not the exporter's timestamp merge. It passes 148 checks, reproducing
all new task fields and all 32 original anchor records exactly. An intentionally
altered history rejects, and retrying an existing output directory rejects
without modifying it. Exact evidence is retained in results/ledger-ml-continuous-022;
the tracked receipt is evidence/ml-continuous-preparation-022.json.

This prepares a more suitable development schedule; it does not launch a paid
comparison, establish benefit or open reserved final outcomes. The incumbent
1.1.9 dynamic evidence adapter and subsequent comparative protocol still need
freezing before inference. Current pilot/evaluation remain unchanged. Main/PyPI
unchanged and the 20% objective remains unmet.
