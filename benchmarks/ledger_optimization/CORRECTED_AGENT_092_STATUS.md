# 092: corrected-history agent trial

Status at 2026-09-14 13:53:30 UTC: **development trial stopped for a confirmed
numerical-budget enforcement defect. No restart. Goal unmet.**

The Gnomon-without-ledger session `item_1372862_store_12/round-2` ran a local
100-configuration ridge grid through `numerical.predict`, with three folds per
configuration, outside the lab counter. The executed loop and successful returned
output imply at least 300 unmetered ridge fits, exceeding the declared 60-attempt
budget; the lab recorded only four attempts. Additional random-forest searches
were also observed. This is a benchmark enforcement defect, not a demonstrated
Gnomon numerical or ledger defect. Earlier statements of “no budget violations”
only checked recorded lab/API counters and were too broad. All previous scores
are now qualified pending an off-lab modelling audit across arms and prior runs.

The affected session made 16 successful API requests, retained a valid seasonal
checkpoint, and did not complete model comparison/selection. It was not replaced
with a fallback or removed from the evidence. All five verified trial processes
were stopped, with no forced kills or tracked processes remaining live. The stop
receipt reports 165 completed sessions; artifacts from interrupted sessions are
mirrored, with 26,285 local files hashed; final cost accounting is pending. Main/PyPI and the final
holdout remain untouched.

Snapshot008, captured before the stop, contains 162 valid forecasts and 161 fully
completed workflows. Its 44,400 recorded numerical/evidence checks passed, as did
20,783 file hashes and 27 frozen source hashes. Those checks did NOT establish
completeness of model-fit accounting. The archive is 461,464,449 bytes, SHA-256
`cce6fc32aa354e6829c841f2846f01695174e662765966ab555a8fcd3add4e1a`.

Diagnostic-only scores for 53 cases completed by all arms (no success filter):

| Arm | Mean RMSLE | Matched reported tokens | Forwarded attempts |
|---|---:|---:|---:|
| Hermes | 0.473854501 | 12,515,339 | 709 |
| Gnomon without ledger | 0.477075188 | 12,937,617 | 704 |
| Ledger | 0.484329018 | 11,677,447 | 656 |

The 1.52% higher ledger error is retained, not promoted as a fair-budget efficacy
comparison. The incomplete session remains in all-session reporting; matched
scores require only that every arm finished that case, not workflow success.
Across the 162 archived sessions, 2,108 forwarded agent attempts include 2,106
successful flash responses, two failures with unknown usage, and 37,939,641
reported tokens. These costs are cumulative with snapshots001–007, not additive.
Pilot, readiness, and the remaining stopped-run costs need separate accounting.

Next: preserve and account for every stopped-run artifact, audit off-lab model
fitting across arms, and enforce or independently account for actual modelling
before freezing a fresh fair trial. Do not repair this run by silently changing
budgets, dropping cases, or rerunning selected failures.

Receipts: `evidence/corrected-agent-092-development-audit-008.json` and
`evidence/corrected-agent-092-unmetered-incident-001.json`.

## Snapshot007 — historical, qualified by the budget defect above

Status at 2026-09-14 12:58:57 UTC: **pilot complete; development evaluation running**.
Snapshot007 captures 142 completed sessions, all valid and workflow-complete.
It passes 37,648 independent numerical/evidence checks and verifies all 17,951
archived files and 27 frozen trial source hashes. The remotely saved and locally
verified archive is 385,186,223 bytes, SHA-256
`fa4ad0caaea29ef96c478c2fa313916d1e2b425859ef19776a64a8ff014bbb73`.

The 47 cases completed by all three arms give:

| Arm | Mean RMSLE | Matched reported tokens | Forwarded attempts |
|---|---:|---:|---:|
| Hermes | 0.464976479 | 11,062,933 | 624 |
| Gnomon without ledger | 0.468758493 | 11,555,257 | 623 |
| Ledger | 0.475862922 | 10,118,182 | 574 |

Ledger is **1.52% worse** than Gnomon without ledger, using approximately 12.4%
fewer reported tokens. Each Gnomon arm has one failed attempt with unknown usage;
reported tokens are incomplete. Matching requires all three arms to have finished
the case regardless of validity; all happen to be successful. The one unmatched
session remains in the archive and total costs. These are two reused development
series, not held-out results.

The 27 matched mature cases (origins >=10) give mean RMSLE 0.442357221 Hermes,
0.451885363 Gnomon, 0.460111729 ledger: ledger is 1.82% worse. Cold and origins-4–9
subsets remain 2.15% and 0.42% worse. The first three matched late cases (origins
>=22, a subset of mature) give 0.521574428 Hermes, 0.531904009 Gnomon, and
0.513726183 ledger: 3.42% lower ledger error. Three late cases do not establish
a phase effect, improvement trend, or the 20% target. No passing claim is made.

All 142 sessions made 1,836 forwarded agent attempts: 1,834 successful flash
responses and the same two previously preserved failures. Reported agent usage
is 32,999,704 tokens, excluding unknown failed-attempt usage. Readiness adds
143 probes and 1,988 reported tokens, retaining the existing local deadline
failure with unknown usage. The previously blocked extra agent request remains
retained and was not forwarded. No new failure, free retry, rerun, score exclusion,
or live trial change was introduced. The audit made zero API/provider calls.

These are cumulative development costs including snapshots001–006; do not add
them together. Pilot costs remain separate. Receipt and reproduction hashes:
`evidence/corrected-agent-092-development-audit-007.json`.
The final holdout remains unopened and the objective remains unmet.

## Snapshot006

Status at 2026-09-14 12:00 UTC: **pilot complete; development evaluation running**.
Snapshot006 captures 120 completed sessions, all valid and workflow-complete.
It passes 28,200 independent numerical/evidence checks and verifies all 14,631
archived files and all 27 frozen trial source hashes. The remotely saved and
locally transferred archive is 284,413,719 bytes with SHA-256
`6d620fe945064f7b363454277458f859da70ea2a7198dfaa8e893aa848ebc664`.

The 39 cases completed by all three arms give:

| Arm | Mean RMSLE | Matched reported tokens | Forwarded attempts |
|---|---:|---:|---:|
| Hermes | 0.460710743 | 9,265,376 | 519 |
| Gnomon without ledger | 0.464556708 | 9,700,024 | 516 |
| Ledger | 0.470243575 | 8,212,738 | 476 |

Ledger is **1.22% worse** than Gnomon without ledger, with approximately 15.3%
fewer reported tokens. Both Gnomon arms have one failed attempt with unknown
usage, so their reported token totals are incomplete. Matching requires all
three arms to have finished the case, regardless of validity; all happen to
be successful. The three unmatched sessions remain in the archive and costs.
These are two reused development series, not held-out results.

The 19 matched mature cases (origins >=10) give mean RMSLE 0.424077338 Hermes,
0.436156168 Gnomon, 0.441945199 ledger: ledger is 1.33% worse than Gnomon.
The cold and origins-4–9 subsets remain 2.15% and 0.42% worse respectively.
There are no late cases (origin >=22). No phase effect, trend, or improvement
claim is established by these partial subsets. The 20% target remains unmet.

All 120 completed sessions made 1,552 forwarded agent attempts: 1,550 successful
flash responses and the same two previously preserved failures. Agent reported
usage totals 27,863,986 tokens, excluding unknown failed-attempt usage. Readiness
adds 121 probes and 1,680 reported tokens, retaining the previously disclosed
local deadline failure with unknown usage. The previously blocked extra agent
request remains retained and was not forwarded. No new failure, free retry,
rerun, score exclusion, or live trial change was introduced in this audit.
No provider or API calls were made for the audit.

These are cumulative development costs including snapshots001–005; do not add
them together. Pilot costs remain separate. Receipt and reproduction hashes:
`evidence/corrected-agent-092-development-audit-006.json`.
The final holdout remains unopened.

## Snapshot005

Status at 2026-09-14 11:08 UTC: **pilot complete; development evaluation running**.
Snapshot005 captures 95 completed sessions, all valid and workflow-complete.
It passes 19,635 independent numerical/evidence checks, verifies all 11,218
archived files, and confirms all 27 frozen trial source hashes. The remotely
saved and locally transferred 194,543,022-byte archive has SHA-256
`ba063f05eb4b6125fd711775b71b32e78f8f2773b950d7e068e4245d4dbb1614`.

The 31 cases completed by all three arms give:

| Arm | Mean RMSLE | Matched reported tokens | Forwarded attempts |
|---|---:|---:|---:|
| Hermes | 0.471409626 | 7,617,081 | 420 |
| Gnomon without ledger | 0.466919210 | 7,820,898 | 412 |
| Ledger | 0.472807055 | 6,578,167 | 385 |

Ledger is **1.26% worse** than Gnomon without ledger, with approximately 15.9%
fewer reported tokens. Both Gnomon arms have one failed attempt with unknown
usage, so reported token totals are incomplete. Matching uses completion
presence, not success or score; all completed cases happen to be successful.
The two unmatched sessions remain in the archive and total cost accounting.
These are still only two reused development series, not held-out results.

The frozen mature subset (origins >=10) now contains 11 matched cases: mean
RMSLE 0.427586260 Hermes, 0.422159192 Gnomon, 0.428588915 ledger, giving ledger
1.52% worse. Cold and origins-4–9 subsets remain 2.15% and 0.42% worse respectively.
No late cases (origin >=22) are present. These partial subsets do not establish
a phase effect, a trend, or the 20% target.

All 95 sessions made 1,246 forwarded agent attempts: 1,244 successful flash
responses and the same two failures previously preserved. Agent reported usage
is 22,523,603 tokens, excluding unknown failed-attempt usage. Readiness adds
96 probes and 1,330 reported tokens, including the previously disclosed local
wall-clock timeout with unknown usage. The existing locally blocked agent
request remains retained and was not forwarded. No reruns, free agent retries,
case exclusions, or live configuration changes were introduced by this audit.
No new provider or API calls were made for the audit.

These are cumulative development totals including snapshots001–004; do not
add snapshots together. Pilot costs remain separate. Reproduction and hashes:
`evidence/corrected-agent-092-development-audit-005.json`.
The final holdout remains unopened and the target remains unmet.

## Snapshot005 coverage diagnostic

An offline diagnostic of all 32 saved ledger sessions passes 409 checks against
the snapshot inventory. The two opening sessions have no history; all 30 later
sessions have one review whose union of comparisons represents every prior
origin. Exact configuration-pair support now reaches nine matched origins.
Eighteen later selected configurations have matching prior evidence in returned
cards; twelve do not. All twelve are also outside the complete eligible
configuration catalogue, so additional pair pages would not supply those exact
configurations. Nine reviews advertise further pair pages, but this does not
explain these twelve unsupported choices.

Among the twelve mature ledger sessions captured in this snapshot, eight select
a configuration with matching prior comparisons and four select one outside
the catalogue. This is a ledger-session coverage diagnostic, not the eleven-case
all-arm matched accuracy subset above. The diagnostic establishes saved evidence
availability only; it does not establish whether the agent used it, a product
bug, or the cause of forecast error. The trial remains unchanged. No database
queries, provider executions or API calls were made. Receipt and script hashes:
`evidence/corrected-agent-092-development-coverage-002.json`.

## Snapshot004

Status at 2026-09-14 10:32 UTC: **pilot complete; development evaluation running**.
Snapshot004 captures 77 completed sessions, all valid and workflow-complete,
including both sessions with agent API errors. It passes 13,929 independent
numerical/evidence checks and verifies all 8,828 archived files against their
inventory. The remotely saved and locally transferred 137,840,124-byte archive
has SHA-256 `b174257ffe261c0894209ca3e6517dca33f198e7608c9989878617fe99e68c7c`.
All 27 trial source hashes match the frozen manifest. No live settings changed.

The 25 cases completed by all three arms give mean RMSLE:

| Arm | Mean RMSLE | Matched reported tokens | Forwarded attempts |
|---|---:|---:|---:|
| Hermes | 0.459180492 | 6,299,423 | 343 |
| Gnomon without ledger | 0.455882680 | 6,326,711 | 333 |
| Ledger | 0.462747145 | 5,270,241 | 311 |

Ledger is **1.51% worse** than Gnomon without ledger. Matching requires all arms
to have finished a case, irrespective of validity; all happen to be successful
here. The two unmatched sessions remain in the archive and cost accounting.
These are still only two reused development series, not held-out results.

Using the frozen phase thresholds, the eight cold cases give ledger 2.15% worse,
the twelve origins-4–9 cases give 0.42% worse, and the first five mature cases
(origins 10–12) give 3.82% worse. Mature RMSLE is 0.313852545 Hermes,
0.313264519 Gnomon, 0.325227598 ledger. There are no late cases (origin 22+) yet.
These small subsets do not establish a phase effect, a trend or the 20% target.

All 77 completed sessions together made 1,011 agent API attempts: 1,009 successful
flash responses and two failures with unknown usage. Reported agent tokens total
18,295,380. Readiness made a separate 78 probes with 1,078 reported tokens: one
probe reached its 30-second wall-clock deadline, logged a local 504 marker and
unknown usage, then the predeclared readiness retry admitted the session. This
is not an established upstream HTTP 504 or a failed agent workflow. Both Gnomon
arms' matched token totals above exclude unknown failed-attempt usage; billed
dollars are unavailable. One additional agent request was blocked locally and
not forwarded. No free agent retry, rerun or case exclusion was introduced.

These totals include prior development snapshots; do not add their costs.
Pilot costs remain separate. Snapshot and reproducible audit receipt:
`evidence/corrected-agent-092-development-audit-004.json`.

## Snapshot003 and subsequent coverage/incident checks

At 2026-09-14 09:48 UTC, the development evaluation was running.
The third snapshot contains 54 completed sessions, all valid and workflow-complete,
and passes 7,787 independent numerical/evidence checks. All 5,978 archived files
match their inventory; the remotely saved archive and its transferred copy share
SHA-256 `e164fb4c7aef20c36b020c38395868ad0a18834f7eca58a3be97b73ba1348885`.
All 27 trial source hashes still match the frozen manifest.

The 17 cases completed by all three arms give mean RMSLE **0.469529765 Hermes,
0.471083977 Gnomon without ledger, 0.473327956 ledger**. Ledger is **0.48% worse**
than Gnomon. Matching here uses completion presence, not success or score; every
completed workflow happened to be successful. The other three sessions remain
in the audit and cost totals. These are two reused series at origins 0–7 and
0–8, with no mature-history cases yet. This does not establish the 20% target.

The failure-aware wire audit retains 715 forwarded attempts: 714 successful
flash responses and the previously recorded proxy failure. Reported agent
usage is 12,960,369 tokens; the failed attempt has unknown usage, so this is
not a complete token total. Readiness adds 54 requests and 756 reported tokens.
A seventeenth request in that same session was blocked locally and never
forwarded. The session still completed its typed workflow within 16 forwarded
requests. No free retry, rerun or score exclusion was introduced.

On the same 17 matched cases, reported tokens are 4,515,422 Hermes, 4,484,226
Gnomon, and 3,342,292 ledger; Gnomon's failed-attempt usage remains unknown.
Billing is unavailable. These are cumulative development costs, including
earlier snapshots, and must not be added to their totals. Pilot costs are separate.
Receipt and reproduction script hashes:
`evidence/corrected-agent-092-development-audit-003.json`.

An offline coverage diagnostic of all 18 ledger sessions in snapshot003 passed
185 checks. All 16 nonopening sessions saved one review with usable history;
each review represents every prior origin somewhere in its returned comparisons.
Exact configuration pairs have at most four matched origins, however. Nine later
selections have matched prior evidence in returned cards and seven do not.
All seven are also absent from the complete eligible configuration catalogue,
which is not paginated: additional pair pages would not supply those exact
configurations. Thus the page limit does not explain these seven selections.
Three reviews advertise additional pages, without a second review in those
session logs. Absence from returned cards is not proof of absence from the ledger.
This diagnoses saved evidence coverage, not whether the agent read or used it,
and does not establish the cause of the score difference. No database queries,
forecasts or API calls were made for this diagnostic, and the live trial is unchanged.
Receipt: `evidence/corrected-agent-092-development-coverage-001.json`.

At the 70-session live check (10:18 UTC), all completed workflows remained valid
and complete with no budget violations. A second API incident occurred in ledger
`item_1047756_store_23`, round 11: request 2 returned HTTP 429 with the message
`per-model concurrency limit exceeded, retry shortly`. It counted against the
same 16-request budget. The session received 15 successful flash responses and
completed 23 numerical attempts and the final workflow, with no additional
correction turn, session rerun or exclusion. Its 294,606 reported tokens exclude
the rejected attempt's unknown usage; no zero-cost or billing assumption is made.
The 48-check wire audit and original failure payload are retained in
`evidence/corrected-agent-092-service-incident-002.json`. This reports a service
concurrency limit, not a demonstrated forecasting defect; it does not identify
the source of other concurrent traffic. Earlier score snapshots remain unchanged.

## Earlier snapshots and incidents

At 2026-09-14 09:24 UTC, the development evaluation was running.
The second development snapshot includes 39 completed sessions and passes 4,677
independent checks, with all sessions complete and valid. Its 13 fully matched
cases (two series, origins 0-5 and 0-6 respectively) give mean RMSLE 0.483503490
for Hermes, 0.481951446 for Gnomon without ledger and 0.484421208 for ledger:
ledger is **0.51% worse** than Gnomon. This is a small completed subset, not a
mature-history or final result. Do not interpret changes from earlier snapshots
as treatment improvement: their case composition differs. The other cases
remain in the run; no settings or completion handling changed.

See `evidence/corrected-agent-092-development-audit-002.json` for retained snapshot
hashes, failures and costs. These 39 sessions consumed 529 agent requests and
9,827,684 tokens. Readiness probes added 39 requests and 546 tokens. All 529 agent
responses identified deepseek-v4.1-flash; the wire audit passed 1,665 checks.
Billed dollars remain unavailable. These cumulative costs include audit001's
19 sessions; do not add the snapshots together. Pilot costs remain separate.

The initial streamed archive download reported exit 0 but failed gzip end-of-stream
validation during extraction. Its bytes and the failure are preserved, and no
scores from that damaged copy were used. A fresh all-completed-session snapshot
was saved remotely, transferred with rsync, and verified against its remote
SHA-256 (`e69e6f22915837f4d9fcbd1ada3f687d6e0770e9b2100f46018351fa0f5bcd97`).
All 4,261 enclosed source files matched their inventory. The recovery reran no
evaluation sessions and made no agent or provider calls.

Audit001 remains in `evidence/corrected-agent-092-development-audit-001.json`:
19 sessions, 1,544 checks; six matched early cases gave ledger 4.6% worse than
Gnomon. Neither provisional snapshot establishes the 20% target.

At the 46-session live check (09:33 UTC), every completed session remained valid
and workflow-complete, with no budget violations. One subsequent Gnomon-control
session (`item_1304243_store_32`, round 7) had a proxy 502/URLError response.
It completed within the same 16-request limit: 15 successful responses, one
failed attempt, 47 numerical attempts, no extra correction turn. The 254,269
reported tokens cover the successful responses; the failed attempt's usage and
upstream billing are unknown, not zero. Its 48-check wire/budget audit and raw
file references are retained in `evidence/corrected-agent-092-service-incident-001.json`.
No session was rerun and no forecast-engine defect was established. Later audits
must include failed responses explicitly; earlier all-success wire checks are
snapshot-specific assertions, not a rule for excluding failures.

The offline `audits/wire_092.py` now handles both successful and failed attempts,
retains incomplete workflows, and reports `usage_complete` and
`attempts_without_usage`. It rejects missing receipts, dropped attempts, changed
model settings and inconsistent usage arithmetic. Six regression tests pass.
It reproduced the prior 529-successful-request snapshot exactly and verified the
actual 16-attempt incident, including its one unknown-usage failure. This audit
change does not alter the frozen trial. Receipt: `evidence/corrected-agent-092-wire-audit.json`.

```sh
python3 -m benchmarks.ledger_optimization.audits.wire_092 /path/to/snapshot --output /tmp/wire-audit-new.json
```

Use a new output path. Readiness probes are separately accounted for; the wire
audit's token totals include only reported usage, never an estimate for missing
responses or billed dollars.

The fresh 312-session development evaluation continues. See the
[complete pilot result](CORRECTED_AGENT_092_PILOT.md) for the separate pilot's
worse ledger accuracy; pilot and development scores are not pooled.
This is development data. The 20% held-out objective remains unproven.

The new checkpoint-v5 trial keeps 13 common numerical, orchestration and tool
source files byte-identical to checkpoint-v4. The ledger treatment combines the
091 production-history correction, the 088/090 recording-visible catalogue, and
087 brief evidence cards. It cannot isolate those three components' effects.
Published Gnomon 1.2.0 executes and stores forecasts; the separately identified
development comparison replaces only history review. Hermes and no-ledger
controls run afresh with the same tools, information availability and budgets.

Sources froze at `a7b50d1`, with the documented task path corrected at `5c05bd0`
before tests or inference. A launch guard first rejected the older runtime task
manifest; no test or paid inference ran in that rejected attempt. The original
continuous task hash was then used, without changing the case selection.

Preflight completed with exit 0 and 29 check groups. The new regression proves
later retrospective fits do not hide the already-scored production comparison,
change its saved evidence, or require new forecasting during review. Common
numerical equivalence, completion/reserve/visibility checks, and native Hermes
execution with synthetic upstream responses passed. There were no paid preflight
API calls. The runtime separately matched all 12,173 retained Hermes archive
files and all 40 Gnomon package files in the pinned 1.2.0 wheel. All 27 tested
trial source files matched the local source hashes.

The pipeline launched at 07:37:22 UTC, PID 3757757, with PID start ticks and boot
identity recorded. Its live process was verified after launch. It runs a fresh
36-session pilot, promoting only if every arm completes at least 11/12 workflows
and integrity checks pass. Promotion starts a separate fresh 312-session
development evaluation. Accuracy is not a pilot promotion criterion; neither
stage opens the final holdout or protected validation055 data.

Model: Engy `deepseek-v4.1-flash`; Gnomon execution/storage: published **1.2.0**.
Native Hermes completion records still say `checkpoint-v4`: that inherited
completion protocol is intentionally unchanged; the outer experiment is 092.

## Evidence and monitoring

* Protocol: `benchmarks/hermes_ml_checkpoint_v5/PROTOCOL.md`.
* Compact launch/archive receipt: `evidence/corrected-agent-092-launch.json`.
* Local frozen preflight archive: `results/corrected-history-092-preflight-bundle.tar.gz`.
* Remote run: `/root/gnomon-ledger-ml-v3/code/results/corrected-history-092-run-001`.
* Remote launch record: sibling `corrected-history-092-launch.json`.
* Remote preflight: sibling `corrected-history-092-preflight-001`.
* Remote launcher logs: sibling `corrected-history-092-setup-002`.

From this repository:

```sh
python3 -m benchmarks.ledger_optimization.pod_history_092 status
python3 -m benchmarks.ledger_optimization.pod_history_092 mirror
```

Status verifies the process identity, not just a lock or status file. Neither
command restarts work. Mirror preserves remote evidence and does not delete
local files. The preflight archive is a frozen preflight/launch snapshot, not a
complete live trial archive. Actual usage, failures, accuracy and uncertainty
must be reported from the completed trial and its independent audit. Main/PyPI
remain unchanged.
