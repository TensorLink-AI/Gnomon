# 092: corrected-history agent trial

Status at 2026-09-14 09:48 UTC: **pilot complete; development evaluation running**.
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
