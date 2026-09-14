# Guarded pilot 093 — completed

Latest audited continuation: 97/312 sessions, 31 matched cases (2026-09-14
17:06 UTC). The pilot is complete; the full development continuation is active.
See the fourth audit below. The 20% target remains unmet.

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
