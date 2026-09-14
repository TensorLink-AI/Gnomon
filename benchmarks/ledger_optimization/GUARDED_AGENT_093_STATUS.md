# Guarded pilot 093 — completed

Latest audited continuation: 208/312 sessions, 69 matched cases (2026-09-14
20:04 UTC). The pilot is complete; the full development continuation is active.
See the thirteenth audit below. The 20% target remains unmet.

All 36 sessions finished. Every forecast is valid; plain Hermes completed 11/12 full workflows, Gnomon without ledger 12/12, ledger 12/12. The frozen completion gate passed. Independent local audit: 4,227 file hashes verified, 2,942 checks, zero integrity failures.

| Arm | Mean per-case RMSLE | Reported tokens | Model requests |
|---|---:|---:|---:|
| plain | 0.512293 | 1,998,383 | 150 |
| gnomon | 0.528709 | 2,343,122 | 164 |
| ledger | 0.497905 | 1,841,347 | 142 |

Ledger versus Gnomon without ledger: 5.83% lower RMSLE and 21.41% fewer reported tokens. The exploratory four-series bootstrap interval for error reduction is -2.18% to +10.54%; it crosses zero. This does not establish the 20% target or a reliable overall accuracy gain.

All 456 model requests returned usage; zero API errors or unknown usage. Readiness probes added 36 requests and 504 reported tokens. Billing dollars are unavailable.

The sole incomplete workflow was plain/item_1304243_store_32/round-0: repeated data summaries exhausted the request budget, leaving a valid seasonal baseline. It remains in the scores. Eight ledger sessions made numerical-evidence queries after outcomes matured. Native memory/skill tool usage is separately recorded in the receipt.

Only first three origins of four reused development series were tested. These are cold-start cases; the pilot cannot measure long-running ledger value. The conditional continuation plan retains every pilot session and all arm state. The paid continuation launched at 2026-09-14 15:49:28 UTC after the completion gate, 41 synthetic integration checks, 1,011 independent resumed-run checks and runtime source re-verification passed. It retains all 36 pilot sessions and runs only the remaining 276; no pilot reruns. First new model request verified. Final holdout remains closed; main/PyPI unchanged.

Authoritative receipt: `evidence/guarded-agent-093-pilot-final.json`. Verified original archive and independent results: `results/guarded-history-093-pilot-final-001`.

Continuation receipt: `evidence/guarded-agent-093-development-launch.json`. Monitor with `python3 -m benchmarks.ledger_optimization.pod_guarded_093 --run guarded-history-093-development-001 --launch guarded-history-093-development-launch-001`. The 36/312 initial completed count consists entirely of retained pilot sessions.

## First continuation audit

Nine additional sessions independently audited: all nine valid and full workflows; 1,159 checks passed, zero integrity failures. Their retained pilot log prefixes and the terminal pilot inventory match. Combined audited evidence contains 45 sessions and 14 all-three matched cases: plain 0.528431, Gnomon 0.559593, ledger 0.510393 mean per-case RMSLE. This partial 8.79% ledger reduction versus Gnomon is development monitoring, not a final or reliable superiority finding. Per-arm incremental totals do not necessarily contain the same cases and must not be compared directly. Receipt: `evidence/guarded-agent-093-development-audit-001.json`.

## Second continuation audit — 2026-09-14 16:13 UTC

Twenty continuation sessions audited, all valid/full: 2,337 snapshot hashes verified, 2,984 independent checks, zero failures or missing shutdown records. All 4,227 original pilot inventory files reverified, and retained pilot experiment prefixes match in every new session. This supersedes the first continuation audit; do not add both batches.

Combined evidence contains 56 audited sessions and 18 all-three matched cases. Matched mean RMSLE: plain 0.533588, Gnomon 0.559279, ledger 0.500957. Matched reported tokens: plain 3,385,740, Gnomon 4,243,690, ledger 3,166,221. Ledger has 10.43% lower error and 25.39% fewer reported tokens versus Gnomon without ledger in this partial set. Numerical attempts are 229/274/294 respectively, so token reduction is not fewer model fits. All outcomes remain included; plain's baseline-only pilot case is retained. No held-out data accessed, no final claim; the 20% target remains unmet. Receipt: `evidence/guarded-agent-093-development-audit-002.json`.

## Third continuation audit — 2026-09-14 16:28 UTC

Thirteen additional, disjoint sessions passed 2,479 independent checks. Combined
with the original pilot and audit-002: 69 audited sessions, 23 matched tasks per
arm. Mean RMSLE is plain 0.497121, Gnomon 0.527513, ledger 0.468983: 11.10% lower
ledger error versus Gnomon without ledger. Matched reported tokens are 4,646,863 /
6,006,112 / 4,174,077 respectively, a 30.50% ledger reduction versus Gnomon. Fits
are 299 / 334 / 370: fewer tokens do not imply fewer numerical executions.

All forecasts are valid. Full workflows: plain 22/23, Gnomon 22/23, ledger 23/23.
Both baseline-only outcomes remain in scores. Each arm had zero native memory or
skill calls and no saved native memory. Native persistence is available and was
synthetically verified, but this run has not demonstrated agents using it.
Ledger evidence was available in 19/23 ledger sessions, including two displaying
recent/lifetime disagreement. The longest audited history has eight past origins;
there is no mature-history (ten-plus origins) efficacy evidence yet. Reading a
card does not establish causal reliance or a causal explanation for improved error.

Transfer caveat: the first SSH archive stream returned exit 0 but its local file
was truncated. Its hash check failed. The pod's unchanged original archive was
rechecked; a second transfer verified each 2 MiB chunk, the full archive and all
1,573 extracted file hashes. Both attempts are retained. No failed copy was used
for analysis. All 4,227 original pilot files and new log prefixes reverified.

Receipt: `evidence/guarded-agent-093-development-audit-003.json`. Audit-003 is a
disjoint delta to audit-002, not a replacement; audit-001 remains superseded by
002. Main/PyPI/live sources unchanged; the 20% target is unmet and final data is
unopened. These partial results cannot establish general ledger superiority.

## Completion and series concentration diagnostic

On the same 23 fully audited matched cases (not the newer live-only counts),
ledger improves 11.10% versus Gnomon overall. Restricting to the 21 cases where
all three arms completed the full workflow gives 10.41%. This outcome-selected
subset is a diagnostic, not a replacement primary estimate or unbiased causal
comparison. The two incomplete-workflow cases contribute 14.1% of the observed
net absolute error difference; the improvement is not solely their contribution.

Most of the current net difference comes from item_1047756_store_23: contribution
0.05294 to pooled difference 0.05853, about 90%. Its ledger error is 16.78% lower
across eight matched origins. Other series show +4.08% (nine origins), +6.76%
(three), and -3.44% (three). Positive means lower ledger error. The unequal
interim series counts and concentration make projection to the complete run
unsafe. Keep the full scheduled grid and all failures; do not reweight or remove
cases to improve the result. Receipt:
`evidence/guarded-agent-093-completion-contrast-001.json`.

## Executed-forecast opportunity diagnostic

Recomputed all 136 current-origin production forecasts across the same 23 audited
matched cases; verified equality of 58 repeated configuration predictions across
executions/arms. Perfect future-aware selection from their union gives mean RMSLE
0.440175, versus Gnomon selected 0.527513: 16.56% reduction. Thus changing selection
among these particular already-executed forecasts cannot achieve the 20% target
on this subset. Ledger's own executed-set hindsight mean is 0.448149, versus its
selected 0.468983. No additional model fits were made.

This is a deliberately future-aware diagnostic, not a policy or a fair extra
arm. It pools forecasts not individually executed by every arm and does not
bound the entire permitted parameter search. A ledger-assisted agent could
potentially discover different configurations within the same allowed tools;
that would require prospective development evidence. The full scheduled run
continues unchanged. Do not stop it, remove cases, change model families or open
the final set on the strength of this partial bound. Receipt:
`evidence/guarded-agent-093-executed-opportunity-001.json`.

## Fourth continuation audit — 2026-09-14 17:06 UTC

Twenty-eight additional disjoint sessions passed 7,173 independent checks, with
zero integrity failures or shutdown gaps. Verified 3,584 snapshot files, all
4,227 original pilot files and 28 retained pilot log prefixes. The original pilot
plus audit-002, 003 and 004 contain 97 sessions and 31 all-three matched cases.

| Arm | Matched mean RMSLE | Matched reported tokens | Full workflows, matched |
|---|---:|---:|---:|
| plain | 0.476790 | 6,623,719 | 30/31 |
| gnomon | 0.511749 | 8,594,138 | 28/31 |
| ledger | 0.452969 | 6,010,877 | 31/31 |

Ledger error is 11.49% lower and reported tokens 30.06% lower than Gnomon without
ledger on this partial development set. Numerical attempts are 405/432/491;
ledger uses more fits despite fewer tokens. All incomplete workflows remain in
scores. Across all 97 sessions, full completion is 32/33, 29/32 and 32/32.

Native memory is now used: plain Hermes saved one note in
item_1047756_store_23/round-10. Its exact text appears in the first forwarded
model request of rounds 11 and 12. Three sessions retaining memory represent one
write and two carried-forward copies, not three writes. Plain made two memory
calls and one skills-list call; Gnomon made one skills-list call but saved no
native memory; ledger made no native memory/skill calls. This supersedes the
earlier observation of no native use. It does not verify the note's factual
claims or establish a causal memory benefit.

Ledger evidence was available in 28 sessions, with up to 13 past origins and two
recent/lifetime disagreements. No held-out targets accessed; live treatment,
main and PyPI unchanged. Receipt:
`evidence/guarded-agent-093-development-audit-004.json`; memory verification:
`evidence/guarded-agent-093-native-memory-001.json`.

## Native memory claim audit — one observed note

The plain-arm note above calls its ranges "matured outcomes", but combines
backtest and matured forecast evidence. The exact Ridge configuration has four
matured forecasts ranging 0.644845–0.821518 RMSLE; the note's 0.51 lower endpoint
comes from a backtest. Seasonal has ten matured forecasts ranging
0.697400–1.455127; the note says 0.89–1.60, with the 1.60 endpoint again coming
from a backtest. Its first review returned 18 backtests and two matured
forecasts out of 139 raw records, explicitly labelled as a limited summary.
The agent also read pages of its previous submissions before writing the note.

All 19 matured forecast metrics independently recompute with zero difference;
their outcome recording times are at or before the current origin. This is a
single observed agent interpretation error, not a numerical or visibility defect
and not an estimated failure rate. It motivates keeping evidence type, matched
support and numerical references attached to any persisted lesson. It does not
prove that structured ledger evidence prevents the same error or improves
accuracy. Live prompts and tools remain frozen. Receipt:
`evidence/guarded-agent-093-memory-claims-001.json`.

## Matched comparison support audit

Across the 32 audited ledger sessions, 28 queried available evidence and 20 had
at least one returned pair with three or more shared lifetime origins. An
additional 1,724 checks verified that returned origin rows contain both expected
provider/revision identities, complete horizons, unique origins and past-only
origin times. Earlier batch audits independently recomputed their scores.

Deduplicating a returned pair within each session leaves 139 pair/session
comparisons: 74 have no lifetime overlap, 32 have one origin, nine have two and
24 have at least three. Recent-four support is sparser: 88 have no overlap.
These counts reuse origins across sessions and cannot be treated as independent
statistical samples. No cross-pair ranking was constructed.

The current display already orders pairs by most recent shared origin. A
candidate for a later frozen experiment is to compress zero-overlap comparisons
into an explicit count with exact retrieval pointers, preserving all supported
comparisons regardless of score or support size. This would reduce empty-card
output without hiding losses or claiming unsupported rankings. It has not been
implemented in the live trial or shown to improve accuracy. Receipt:
`evidence/guarded-agent-093-pair-support-001.json`.

## Fifth continuation audit — 2026-09-14 17:28 UTC

Fourteen new sessions passed 5,027 independent checks, zero integrity failures
and zero shutdown gaps. Verified the full archive and 1,999 snapshot files,
all 4,227 original pilot files and 14 retained pilot prefixes. Combined audited
evidence now contains 111 sessions and 36 all-three matched cases.

| Arm | Matched mean RMSLE | Matched reported tokens | Full matched workflows |
|---|---:|---:|---:|
| plain | 0.477626 | 8,094,906 | 35/36 |
| gnomon | 0.506410 | 10,380,791 | 33/36 |
| ledger | 0.456414 | 7,463,953 | 36/36 |

Ledger is 9.87% lower in error and uses 28.10% fewer reported tokens versus
Gnomon without ledger on this partial set. All incomplete workflows are included.
Numerical attempts are 490/505/564: ledger still uses more numerical fits.
Across all audited sessions, full completion is 36/37, 33/36 and 38/38; all
forecasts are valid. Counts outside the matched table cover different tasks.

The existing plain-arm native note persists in five sessions; this is not five
memory writes. Ledger evidence is available in 34 sessions, with up to 16 past
origins. No future/outcome-based changes were made to the running treatment.
The sparse-display prototype is separate and undeployed. Final holdout remains
closed, main/PyPI unchanged, and the 20% target unmet. Receipt:
`evidence/guarded-agent-093-development-audit-005.json`.

## First multi-case mature-history diagnostic

Using the frozen original analyzer's definitions, the 36 audited matched cases
include 14 cold cases (round < 4), ten mature cases (round >= 10), and no later
cases (round >= 22). Later is a subset of mature, not a separate exclusive bin.
All failures remain in each phase's matched denominator.

| Phase | Matched cases | Gnomon RMSLE | Ledger RMSLE | Ledger error reduction |
|---|---:|---:|---:|---:|
| Cold | 14 | 0.559593 | 0.510393 | 8.79% |
| Mature | 10 | 0.412364 | 0.385909 | 6.42% |

There is no observed growth in the aggregate advantage yet. Mature coverage
contains only two series (four and six origins); cold coverage contains all four
series. Different origins and incomplete series coverage prevent a causal
interpretation of the phase difference. Mature full workflows are 10/10 plain,
8/10 Gnomon and 10/10 ledger; these incomplete Gnomon cases remain scored.
Continue the complete scheduled grid unchanged. Do not discard cold starts,
change the mature threshold or claim an accumulated-history benefit from this
partial diagnostic. Receipt: `evidence/guarded-agent-093-phases-001.json`.

## Sixth continuation audit — 2026-09-14 17:51 UTC

Fourteen additional disjoint sessions passed 4,510 independent checks with zero
integrity failures and zero shutdown gaps. The 38,241,006-byte archive, all 2,031
snapshot files, 4,227 original pilot files and 14 retained pilot prefixes passed
hash/prefix verification. Original outcomes and all incomplete workflows remain
in the cumulative analysis: 125 audited sessions, 41 all-three matched cases.

| Arm | Matched mean RMSLE | Matched reported tokens | Model requests | Full matched workflows |
|---|---:|---:|---:|---:|
| plain | 0.483878 | 10,360,044 | 558 | 40/41 |
| gnomon | 0.507830 | 11,926,787 | 585 | 38/41 |
| ledger | 0.462619 | 8,592,963 | 492 | 41/41 |

Ledger has 8.90% lower error and 27.95% fewer reported tokens than Gnomon without
ledger. Numerical attempts remain higher: 569/593/646 respectively. Across all
audited sessions, full completion is 40/41 plain, 39/42 Gnomon and 42/42 ledger;
all forecasts are valid. These unbalanced totals are not the matched comparison.

The one plain-arm native note is retained in seven sessions; no new memory
writes were observed. Ledger evidence was available in 38 sessions, with at
most 18 past origins. Later-phase origins (22+) are not yet represented by this
audit. No change to live sources, main, PyPI or final-data access. Receipt:
`evidence/guarded-agent-093-development-audit-006.json`.

## Readiness timeout and recovered session

Before ledger/item_1304243_store_32/round-19, a task-free readiness probe returned
HTTP 504 after 30.01 seconds, with no reported usage. The second probe passed
before agent execution. The unchanged session then completed its full workflow:
14/14 forecast-call responses, 333,196 reported tokens, zero forecast API errors
and zero missing forecast usage. Readiness remains separately counted: two
requests, 14 reported tokens from the successful probe, one unknown-usage probe.
Unknown usage is not zero. The production cost helper correctly preserves this
distinction; its frozen source hash matches. No run restart, extra agent budget
or score substitution occurred. Receipt:
`evidence/guarded-agent-093-readiness-incident-001.json`.

## Seventh continuation audit — 2026-09-14 18:13 UTC

Thirteen new, disjoint sessions passed 4,643 independent checks, zero integrity
failures and zero missing shutdown records. Verified the 38,504,251-byte archive,
all 1,911 snapshot files, 4,227 original pilot files and 13 retained pilot log
prefixes. The batch includes the completed readiness-timeout recovery described
above. Combined audited evidence has 138 sessions, all 46 cases matched across
the three arms, with every incomplete workflow retained.

| Arm | Mean RMSLE | Reported tokens | Model requests | Full workflows |
|---|---:|---:|---:|---:|
| plain | 0.472122 | 11,660,653 | 626 | 45/46 |
| gnomon | 0.490918 | 13,258,494 | 652 | 43/46 |
| ledger | 0.451259 | 9,599,332 | 545 | 46/46 |

Ledger error is 8.08% lower and reported token use 27.60% lower than Gnomon
without ledger. Numerical attempts are 645/669/719; this remains fewer tokens
with more fits. All forecasts are valid. The plain-arm note persists across nine
sessions, with no additional native-memory write. Ledger evidence is available
in 42 sessions, with up to 20 past origins. Later-phase origins are still absent
from this audit. No live changes or final-data access. The 20% objective remains
unmet. Receipt: `evidence/guarded-agent-093-development-audit-007.json`.

## Updated phase diagnostic — 138 audited sessions

The frozen phase definitions applied to the original pilot and disjoint audits
002–007 give 14 cold matched cases (round < 4), 20 mature matched cases
(round >= 10), and no later matched cases (round >= 22). Ledger RMSLE is
8.79% lower than Gnomon without ledger in the cold subset and 3.41% lower
in the mature subset. Mature mean RMSLE is plain 0.421201, Gnomon 0.423757,
ledger 0.409305; full workflows are 20/20, 18/20, 20/20 respectively, with
every outcome retained in the scores. Mature coverage currently contains
only two of four series, and different phases contain different origins.
This does not identify a causal accumulation effect or support a claim
that the ledger advantage increases with history. The earlier phase
receipt is preserved; this is an expanded diagnostic, not a new experiment.
Receipt: `evidence/guarded-agent-093-phases-002.json`. Live run and final
holdout are unchanged; the 20% target remains unmet.

## Eighth continuation audit — 2026-09-14 18:31 UTC

Eleven additional disjoint sessions passed 4,524 independent checks, with zero
failures or missing shutdown records. All 1,704 copied file hashes and the
whole 37,721,965-byte archive match. All 4,227 original pilot files and the
11 retained experiment prefixes were reverified.

Combined evidence now contains 149 audited sessions and 49 all-three matched
cases. Mean RMSLE is plain 0.470276, Gnomon 0.489343, ledger 0.450318, a 7.97%
ledger reduction versus Gnomon without ledger. Matched reported tokens are
12,675,534 / 13,900,263 / 10,260,890, respectively: 26.18% fewer for ledger.
Matched full workflows are 48/49, 46/49, 49/49. All incomplete outcomes remain
in the numerical scores. Numerical attempts are 681/714/767; token savings
do not imply fewer fits.

The first complete round-22 three-arm case is present (item_1304243_store_32).
RMSLE is plain 0.396421, Gnomon 0.394271, ledger 0.402568: ledger is worse
on this one case. One origin from one series does not establish a late-phase
effect. Ledger evidence was available in
46 of its 50 audited sessions, up to 22 prior origins. Plain Hermes retains
its native-memory note in 11 sessions; persisted copies are not new writes.
The 20% target is unmet and the final holdout remains closed.
Receipt: `evidence/guarded-agent-093-development-audit-008.json`.

## Expanded executed-forecast opportunity diagnostic — 49 matched cases

Recomputed all 290 production forecasts already executed across the audited
49 matched cases, verifying 127 repeated configurations have identical points.
The future-aware union minimum is mean RMSLE 0.428992, only 12.33% below
Gnomon's selected 0.489343. Within-arm hindsight means are plain 0.452718,
Gnomon 0.467346 and ledger 0.435100. This includes failures and uses the
original fixed targets; it neither runs new forecasts nor opens final data.

This union uses future outcomes and predictions from unequal arm-specific
searches. It is not an executable policy or a fourth comparison arm. It bounds
only selections among these already executed predictions, not the entire
permitted configuration space or the unfinished development run. On this
subset, merely choosing better among existing predictions cannot meet 20%.
A subsequent ledger candidate must therefore be evaluated for whether it
helps the agent discover better configurations using the same permitted
model families and budgets; display compression alone is insufficient evidence
of progress toward the numerical objective. Do not modify the active runner
or change the frozen final target in response to this diagnostic.
Receipt: `evidence/guarded-agent-093-executed-opportunity-002.json`.

## Ninth continuation audit — 2026-09-14 18:53 UTC

Twelve disjoint new sessions passed 5,120 independent checks, zero failures
or missing shutdown records. All 1,837 snapshot file hashes, the whole
43,540,000-byte archive, 4,227 original pilot files and 12 retained pilot
experiment prefixes were verified. All outcomes are preserved.

Across 53 matched cases, mean RMSLE is plain 0.470273, Gnomon 0.487109,
ledger 0.451556: a 7.30% ledger reduction. Matched reported tokens are
13,788,858 / 14,891,549 / 11,047,158 (25.82% fewer for ledger), and full
workflows are 52/53, 50/53, 53/53. Numerical attempts are 740/770/847.
The complete audited inventory has 161 sessions; unequal per-arm counts
are not used as a matched performance comparison.

The expanded phase diagnostic has 27 mature matched cases (round >= 10),
with 2.98% lower ledger RMSLE. Its first three later cases (round >= 22)
all belong to item_1304243_store_32: plain 0.347320, Gnomon 0.344929,
ledger 0.344582, only 0.10% lower ledger error. Later reported tokens are
657,013 / 476,367 / 572,271, so ledger uses more tokens than Gnomon on
this small subset. Mature and later overlap; their series/origin coverage
differs from the cold phase. Neither a growing accumulation benefit nor
a population-level late-phase conclusion is established.

Receipts: `evidence/guarded-agent-093-development-audit-009.json` and
`evidence/guarded-agent-093-phases-003.json`. The 20% target remains unmet;
no final targets, active-run sources, main or PyPI were changed.

## Live diagnostic — first resumed third-series Gnomon workflow

The stored grade for gnomon/item_1372862_store_12/round-3 reports a valid
seasonal checkpoint but an incomplete workflow: 16 requests exhausted,
one distinct backtested configuration, four numerical attempts and no
selection after comparison. All requests succeeded and usage is complete
(258,928 reported tokens). Its forecast RMSLE is zero, so the incomplete
workflow is not an inaccurate forecast or an engine failure. It remains
in the matched scores. This adds one incomplete workflow to the four
previously retained cases. Raw grade inspection is preserved separately
from the pending independent session audit; no run changes were made.
Receipt: `evidence/guarded-agent-093-failure-monitor-002.json`.

## Tenth continuation audit — 2026-09-14 19:11 UTC

Twelve new sessions passed 3,756 independent checks, zero failures or missing
shutdown records. All 1,735 snapshot hashes, the 32,062,842-byte archive,
4,227 original pilot files and 12 retained pilot prefixes were verified.
Combined evidence contains 173 sessions and 57 matched cases. Mean RMSLE:
plain 0.477798, Gnomon 0.489355, ledger 0.461857 (5.62% lower ledger error).
Matched tokens: 15,123,743 / 15,807,154 / 12,037,062 (23.85% fewer for ledger).
Matched full workflows: 56/57, 53/57, 57/57. All outcomes remain scored.

Two resumed third-series sessions were incomplete after exhausting the
16-request budget: Gnomon round 3 and plain round 4, both retaining valid
seasonal forecasts with zero RMSLE. The latter is not yet an all-three
matched case in this snapshot. The complete audited inventory contains
six incomplete workflows (plain 2, Gnomon 4, ledger 0).

### Local audit reporting correction

The frozen auditor crashed while summarizing this incremental batch because
a bootstrap resample had zero control RMSLE. Its original stderr and exit
status are retained. A separate local `analyze_guarded_093.py` preserves
all session checks, numerical scores and the original sampling procedure,
but counts undefined relative draws and returns a null relative interval
if any occur. It also reports absolute differences as diagnostics, never
as a substitute target. No draw is dropped or assigned a synthetic gain.
This batch has 69 undefined relative draws in its ledger/Gnomon contrast.
Five regression tests cover positive-data parity, mixed and all-zero control,
retaining losses and invalid input. The preceding real audit was recomputed
to check that its session fields and existing contrasts remain identical.

The frozen live analyzer and all runner sources remain unchanged. Subsequent
local incremental audits should use the new local analyzer, with its source
hash retained in their receipts. No forecast was rerun, no outcomes excluded
and no final holdout opened. The 20% target remains unmet.
Receipt: `evidence/guarded-agent-093-development-audit-010.json`.

## Eleventh continuation audit — 2026-09-14 19:32 UTC

Twelve disjoint new sessions passed 3,024 independent checks with no audit
failures or missing shutdown records. All 1,648 snapshot hashes, the whole
28,537,812-byte archive, 4,227 original pilot files and 12 retained pilot
prefixes were verified. The local zero-control-aware auditor was used and
its hash retained; the frozen live source remains unchanged.

The combined 185 audited sessions contain 61 all-three matched cases. Mean
RMSLE is plain 0.470187, Gnomon 0.483778, ledger 0.457096: 5.52% lower
ledger error. Matched tokens are 16,374,508 / 16,791,982 / 13,005,663
(22.55% fewer for ledger), full workflows 59/61, 56/61, 61/61 and numerical
attempts 845/883/946. All outcomes remain included. The complete inventory
retains seven incomplete workflows: plain two, Gnomon five, ledger zero.
The new Gnomon round-5 third-series budget failure is independently audited.

Expanded frozen-phase results: cold 15 matched cases, 6.99% lower ledger
error; mature 31 cases, 1.43% lower; later seven cases, 3.77% HIGHER ledger
error. Later RMSLE is plain 0.551889, Gnomon 0.538083, ledger 0.558356.
Later tokens are 2,308,988 / 1,339,976 / 1,623,899, also higher for ledger
than Gnomon. Those seven later cases cover only the first two series, and
all complete the workflow. This contradicts a claim of established
late-history benefit in the observed subset; it does not establish a
general negative effect or causally isolate accumulation. Phases overlap
and cover different origins. Preserve the unfavorable evidence.

Receipts: `evidence/guarded-agent-093-development-audit-011.json` and
`evidence/guarded-agent-093-phases-004.json`. The 20% target is unmet and
final holdout access remains closed. No main/PyPI or active-run changes.

## Twelfth continuation audit — 2026-09-14 19:48 UTC

Eleven additional sessions passed 2,969 independent checks with zero failures or
missing shutdown records. Verified the complete archive, 1,414 extracted files,
all 4,227 original pilot files and every retained pilot log prefix. Combined
evidence: 196 audited sessions, 65 all-three matched cases. RMSLE: plain
0.465816, Gnomon 0.479740, ledger 0.454924. Ledger has 5.17% lower error and
22.06% fewer reported tokens than Gnomon without ledger. Full workflows are
63/65, 60/65 and 65/65; every forecast is valid and every case remains scored.
Numerical attempts are 916/978/1002, so token savings do not mean fewer fits.
Receipt: `evidence/guarded-agent-093-development-audit-012.json`.

Both initial series now have all 26 matched origins audited. Under the frozen
phase definitions, their 32 mature cases give only 1.36% lower ledger error;
their final eight cases give 3.14% HIGHER ledger error. All eight latter
workflows completed. This preserves unfavorable evidence and does not support
a growing-history accuracy claim. Phases contain different origins; they do
not identify a causal accumulation effect. The other two series are unfinished.
Receipt: `evidence/guarded-agent-093-phases-005.json`.

A subsequent live grade records another incomplete plain-Hermes workflow at
item_1457251_store_14 round 4: 16 model requests, zero API errors, four numerical
attempts and a single backtested configuration. Its valid seasonal checkpoint
RMSLE is 0.684528 and remains included. This brings observed incompletions to
eight (plain three, Gnomon five, ledger zero). This new grade still needs full
session audit. Receipt: `evidence/guarded-agent-093-failure-monitor-003.json`.

The 20% objective remains unmet. No final data accessed or live sources changed.

## Expanded executed-forecast bound — 65 audited matched cases

Independently recomputed 382 already-executed production forecasts and verified
166 repeated configuration predictions agree. A future-aware choice from the
union across all arms yields RMSLE 0.425652, only 11.27% lower than the matched
Gnomon control (0.479740). Ledger's own executed-set hindsight mean is 0.435550
versus its selected 0.454924. This is not an executable policy: it uses future
outcomes and candidates unequally available to individual arms. It does not
bound the entire allowed configuration space or the unfinished experiment.

Nevertheless, selection-only changes cannot reach 20% on this already-executed
subset. A prospective infrastructure candidate must support better exploration
within unchanged common models and budgets, as well as faithful evidence use.
Do not change the live run or promote the sparse-display prototype on this
result. No new fits, API calls or final data access. Receipt:
`evidence/guarded-agent-093-executed-opportunity-003.json`.

## Thirteenth continuation audit — 2026-09-14 20:04 UTC

Twelve new sessions passed 2,073 independent checks, zero failures or missing
shutdown records. The archive and all 1,463 extracted file hashes verified;
4,227 original pilot files and all new retained pilot prefixes reverified.
Combined evidence: 208 sessions, 69 all-three matched cases. Mean RMSLE is
plain 0.464389, Gnomon 0.475823, ledger 0.452283: 4.95% lower ledger error.
Matched reported tokens are 18,519,814 / 19,197,225 / 14,895,320, a 22.41%
ledger reduction versus Gnomon. Full workflows: 66/69, 64/69, 69/69. All
forecasts remain valid and all cases remain scored. The eighth observed
incomplete workflow (plain item_1457251 round 4) is now independently audited.
Receipt: `evidence/guarded-agent-093-development-audit-013.json`.

Plain Hermes also used native memory on item_1457251_store_14 round 6: one
skills-list request, one malformed memory request rejected, then one successful
memory add. Its saved note exactly matches the add content. This is a second
series with a native note, not proof of correct interpretation or improvement.
Subsequent restoration remains to be checked. Receipt:
`evidence/guarded-agent-093-native-memory-002.json`.

The collection audit on the earlier 65 matched cases found 162 ledger-arm
configuration/origins with complete current backtests but no production forecast.
These cannot become historical forecast evidence retrospectively. Common
prospective collection could increase paired coverage while charging all fits
to every arm. See `PAIRED_COLLECTION_096.md` and
`evidence/guarded-agent-093-collection-opportunity-001.json`. It is an undeployed
proposal, with no new fits or efficacy result. The live run remains unchanged,
the queued seed integration remains pending, and the final holdout stays closed.
