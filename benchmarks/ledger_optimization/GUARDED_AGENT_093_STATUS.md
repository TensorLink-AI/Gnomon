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
