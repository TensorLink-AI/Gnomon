# 092: corrected-history agent trial

Status at 2026-09-14 09:24 UTC: **pilot complete; development evaluation running**.
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
