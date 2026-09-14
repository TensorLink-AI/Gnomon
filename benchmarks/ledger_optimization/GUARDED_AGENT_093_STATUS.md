# Guarded pilot 093 — completed

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
