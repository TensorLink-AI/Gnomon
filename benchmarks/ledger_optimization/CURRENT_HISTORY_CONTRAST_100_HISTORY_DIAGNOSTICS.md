# Candidate 100: accumulated history and search diagnostics

## Updated snapshot: 186 audited sessions, 61 matched cases

The fourth audit expands the descriptive analysis without changing its rules.
Two series now have all 26 matched origins; their cumulative ledger changes
versus no-ledger are 2.62% worse (item 1047756/store 23) and 0.19% better
(item 1304243/store 32). The remaining series have five and four matched
origins. This still supplies no consistent accumulating advantage; neither
completed series establishes the 20% objective.

| Arm | Mean complete CV configurations | Sessions testing RF | Selected final CV minimum |
|---|---:|---:|---:|
| Hermes | 2.66 | 12/61 | 61/61 |
| Hermes + Gnomon | 2.92 | 7/61 | 57/61 |
| Hermes + Gnomon + ledger | 2.74 | 11/61 | 53/61 |

All arms tested Ridge and seasonal models in every matched case. The two
Gnomon arms explored identical complete sets in 4/61 cases and selected the
same configuration in 29/61. These observed differences do not isolate a
causal ledger effect on search or selection. The all-case and per-origin
results, final-CV comparisons and input hashes are retained under
`results/contrast-100-history-progress-003/`. The original snapshot below
remains intact. Diagnostic 104 separately asks about hindsight opportunity
among already executed configurations; it is not a deployable policy.

## Earlier snapshot: 93 audited sessions

These are post-hoc descriptions of 93 independently audited development
sessions, with 30 complete three-arm cases. They do not change the running
protocol, choose a mature-history cutoff, or open final data. Gnomon is the
pinned published 1.2.0 build, with DeepSeek v4.1 Flash in all arms.

## Does the observed advantage grow with history?

Not clearly in this snapshot. The primary all-matched means are 0.480151 for
Gnomon without ledger and 0.493400 for ledger, a 2.76% disadvantage. The two
series with later observations have 12 matched origins each; the remaining two
still have only their three pilot origins. Their cumulative curves do not show
an established accumulating advantage. Later rounds therefore must not be
compared with the four-series pilot as though the population were constant.

`contrast_history_progress_100.summarize` reports each prior-outcome count and
per-series cumulative results. All matched failures remain included; missing
arms stay explicitly pending. Missing origins, duplicates, mismatched origins
and invalid scores reject. A zero control mean makes relative improvement
undefined, not zero or an invented benefit. The frozen plan did not specify a
numeric cold/mature threshold; none is introduced here. The legacy analyzer's
convenience cold/mature labels are not a preregistered decision rule.

The standalone plot is retained as
`results/contrast-100-history-progress-001/history-progress-002.png` and `.svg`.
Each point uses every matched origin through that round for its series. Calendar
conditions change alongside history length; this is not a causal memory curve.

## Search behavior across every matched case

`contrast_exploration_100.exploration` reads authenticated audit rows and their
audited final current-CV tables. It recomputes configuration overlap and selected
CV rank, rather than accepting an agent's narrative rank. These are end-of-session
tables; a configuration in them is not assumed to predate an earlier decision.

| Arm | Mean complete CV configurations | Sessions testing RF | Selected final CV minimum |
|---|---:|---:|---:|
| Hermes | 2.70 | 6/30 | 30/30 |
| Hermes + Gnomon | 2.87 | 4/30 | 28/30 |
| Hermes + Gnomon + ledger | 2.87 | 5/30 | 27/30 |

All arms tested Ridge and seasonal configurations in all 30 cases. Gnomon and
ledger tested identical complete configuration sets in only 3/30 cases, and
selected the same exact configuration in 15/30. All selected configurations
appear in the final complete-CV tables. Thus the outcome difference mixes
configuration exploration with selection among explored candidates. The data
do not support a general claim that ledger tested fewer configurations, or
that following historical evidence necessarily caused a particular loss.

Three largest observed ledger-minus-control errors were retained as explicitly
post-hoc illustrations, with original public decision summaries and hashes:

| Series / round | No-ledger selection and RMSLE | Ledger selection and RMSLE |
|---|---|---|
| item 1047756, store 23 / 4 | Ridge 365/28/alpha10: 0.902152 | Seasonal 7: 1.192336 |
| item 1372862, store 12 / 2 | Ridge 365/14/alpha10: 0.241676 | Ridge 730/28/alpha1: 0.310325 |
| item 1047756, store 23 / 9 | RF 365/28/depth8: 0.999932 | Ridge 365/14/alpha10: 1.057293 |

Each was a valid execution without fallback. In round 9 the control tested RF
and ledger did not; this is an example of different search, not evidence of an
execution failure. Agent explanations are retained as explanations, not treated
as verified causal or business facts. The three cases were chosen after seeing
losses and cannot establish the frequency of any behavior.

## Consequence for the next iteration

Do not change candidate 100 while it runs or dispatch a new paid variant merely
because of these interim losses. The follow-up below measures temporal coverage
of retrieved comparisons before designing a new ledger presentation. It shows
why origin overlap alone cannot classify evidence as redundant. A proposed change must keep
the common model space, raw evidence access and budgets, be frozen before paid
evaluation, and preserve this negative development result.

### Historical coverage follow-up

Among the 30 matched ledger sessions, 26 displayed at least one pair with
historical evidence; 16 displayed a pair with origins outside the current CV
window. Taking the latest available comparison for each displayed pair and
deduplicating windows with identical origins gives 66 session/pair/cohorts:
24 wholly within current CV dates and 42 containing additional dates. These
counts repeat underlying observations across pairs and sessions; they are not
independent sample sizes. They describe what was displayed, not everything
retrievable from storage or what influenced the agent's choice.

The 264 overlapping model/origin score comparisons have a maximum absolute
RMSLE difference of 0.024436. A read-only payload comparison explains an important
distinction: same configuration and origin need not mean the same request.
For item 1047756/store 23 at origin September 13, the round-4 CV execution of
Ridge(window730, lags28, alpha1) received 702 history rows; the original production
execution received 730. Cutoff, target timestamps, future covariates, series and
unit match. The shorter history and its past covariates exactly match the suffix
of the longer history. Their request fingerprints differ. The common predictor
uses the available suffix capped at `window`, so these are different effective
training sets despite the same maximum-window configuration. No numerical
rerun was needed to establish the request difference, and the database hash
was unchanged after inspection. This is not evidence of a ledger scoring bug.

Consequently, a future ledger comparison should distinguish **outside-CV
origins**, **overlapping origins**, and **matching effective training inputs**.
It must not merge or discount historical evidence solely because dates overlap.
Current-CV effective training lengths should be disclosed equally to all arms;
historical training provenance belongs alongside the ledger's stored executions.
This is a proposed infrastructure clarification, not a dispatched candidate or
proof it will improve selection. Next assess whether current comparison pairs
cover the selected configurations and whether historical disagreements concern
comparable training inputs before proposing a changed retrieval rule.

The scripts authenticate prior audited annotation/report hashes, content-addressed
comparison payloads and referenced review payloads before counting. Timestamp
comparison normalizes timezone and precision. The original and follow-up
programs, stdout/stderr, exit records, per-session results and input hashes are
retained in `results/contrast-100-history-novelty-001/`. Compact receipt:
`evidence/contrast-100-history-novelty-001.json`.

Eight regression tests pass for all-case retention, pending groups, origin
validation, order invariance, undefined ratios, configuration overlap, exact CV
ties, absent selections and malformed tables. Running the new analysis on all
93 rows reproduces the previous matched metric exactly. Analysis and test logs,
source hashes and full diagnostic outputs are under
`results/contrast-100-history-progress-002/`; original plots and decision
illustrations are under `results/contrast-100-history-progress-001/`.
Compact committed data: `evidence/contrast-100-history-progress-001.json`.
Audit and source receipt: `evidence/contrast-100-development-audit-003.json`.

No forecasting policy changed, no additional provider or Engy calls were made,
and the reserved M5 and other final targets remain unopened. The 20% objective
and final uncertainty requirement remain unestablished.
