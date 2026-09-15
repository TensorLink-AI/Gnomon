# Candidate 100: completed development pilot

The 36-session pilot completed with all forecasts valid, all workflows complete,
and no API errors. Its independent re-audit passed 32,914 checks, including the
current-CV table and accumulated-evidence annotation checks. The recomputed rows
exactly match the retained report. This establishes the tested interface and
completion behavior; the accuracy result is negative for ledger on this sample.

Published Gnomon 1.2.0, Hermes and Engy deepseek-v4.1-flash were used for the
three-arm comparison. The same four reused Favorita development series, first
three origins and requested seed 7 give 12 matched cases. All arms receive the
already-computed current-CV table. Ledger additionally contrasts that table with
its last requested historical review. Tools, numerical budgets, task data and
completion rules remain fixed.

| Arm | Valid/full workflows | Mean per-case RMSLE | Reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|---:|
| Hermes | 12/12 | 0.497228 | 1,470,749 | 105 | 144 |
| Hermes + Gnomon | 12/12 | 0.503353 | 1,431,909 | 103 | 128 |
| Hermes + Gnomon + ledger | 12/12 | 0.509034 | 1,417,739 | 102 | 132 |

Ledger error is 1.1286% higher than no-ledger Gnomon. The frozen exploratory
series-bootstrap 95% interval for relative improvement is [-2.1789%, -0.4513%]:
one win, five losses, six ties. This is adverse evidence for this pilot, not
evidence of a benefit. Four reused series and one seed do not establish general
performance. These are cold-start origins with at most two prior outcomes; the
pilot contains no mature-history window. The final 20% target remains unmet.

All 310 agent requests and responses are retained, with 4,320,397 reported tokens.
There were another 36 readiness requests and 504 tokens. No API errors, orphan
responses or missing usage were found. Dollar billing remains unknown. Re-audits
and archive reads made no additional provider or Engy calls. Earlier partial
snapshots are subsets of these sessions and must not be added to their costs.

The controller terminated at 2026-09-15 07:29:08 UTC, exit 0. The 23,595,964-byte
archive has SHA-256
`f36ee81f8229ec88a8c8c9c84b2ed9ac7e110c79c8bad420524d16a437fb8737`.
All 4,580 inventoried files verified locally; every chunk passed its first read.
The independent analyzer passed on its first execution. Evidence is retained in
`results/contrast-100-final-001/`, with compact hashes and results in
`evidence/contrast-100-final-001.json`.

The prospectively frozen completion gate passed independently of accuracy.
The gated continuation was dispatched at 09:32:24 UTC after independent local
audit and repeated pod checks of terminal state, exact sources, runtime, plan
and continuation proof. It retains these 36 sessions once and adds 276 later
sessions; it does not restart the pilot. The negative pilot results remain in the
combined comparison. No final-data access, merge to main or PyPI release occurred.

At 09:34:55 UTC, controller PID 173958 and child PID 173989 were confirmed live
by boot/start identities. The accepted-launch receipt binds the exact plan and
continuation proof. All 3,863 copied canonical pilot-session files matched the
terminal inventory; all original pilot files remained unchanged. Two additional
sessions had completed and 28 new agent requests had been forwarded. These are
live counts, not a completed continuation or independently audited new scores.
Receipt: `evidence/contrast-100-continuation-launch-001.json`.
