# Common workflow-progress 097: development pilot results

Published Gnomon 1.2.0, Hermes, Engy deepseek-v4.1-flash. Four reused Favorita development series, three origins, 12 matched cases. This pilot tests completion and early accuracy; it does not establish accumulated ledger value or final efficacy.

| Arm | Full workflow | Mean RMSLE | Tokens | API requests | Fits |
|---|---:|---:|---:|---:|---:|
| gnomon | 12/12 | 0.501086 | 1,565,731 | 121 | 136 |
| ledger | 12/12 | 0.498361 | 1,564,855 | 119 | 156 |
| plain | 12/12 | 0.503288 | 1,624,834 | 121 | 148 |

Ledger vs Gnomon: 0.54% relative RMSLE reduction. Exploratory series-bootstrap 95% interval: [-0.021009705948197732, 0.044289054198687006]. Only four reused series and one requested seed; not confirmatory.

Completion gate passed: True. Gate uses completeness and integrity, not accuracy. Independent checks: 8135; failures: 0; shutdown gaps: 0.

The previous 096 pilot completed 10/12 plain, 11/12 Gnomon, and 12/12 ledger workflows. The new reminders apply equally to all arms without adding fits, requests, task facts or future outcomes. Descriptive per-arm changes are retained in the evidence receipt; do not interpret a before/after improvement as a ledger-only effect.

All raw evidence is in `results/workflow-097-final-001/original`; its archive and every file were hash-verified. All sessions and costs are retained. A continuation, if launched, will retain these 36 sessions once and add 276; do not add pilot costs again to the combined total.

The untouched final evaluation remains closed. The 20% final-evaluation objective is not established.

The continuation was subsequently launched at 2026-09-15 01:08:30 UTC after repeating the terminal gate checks on the pod. Controller/child identity and unchanged retained session files were verified; see `evidence/workflow-097-continuation-launch-001.json`. It adds 276 sessions to the 36 retained here, with the same frozen settings.

## Paired evidence availability

A read-only audit compared the first three origins across the same four reused series in runs 093, 096 and 097. Only pages actually returned to the ledger agent were counted; duplicate pairs within a session were counted once, and inherited prior-origin pages were excluded. Referenced full files matched their hashes and reported horizon/origin counts.

| Run | Displayed pair/session incidences | With matched evidence | Without matched evidence | Matched pair/origin incidences |
|---|---:|---:|---:|---:|
| 093 selected forecasts | 12 | 10 | 2 | 12 |
| 096 prospective collection | 53 | 52 | 1 | 64 |
| 097 collection + common reminders | 42 | 42 | 0 | 54 |

These are overlapping evidence incidences, not independent samples. Exploration and review requests also differed between runs, so this is not a controlled estimate of the collection change alone. It confirms that the current run supplies substantially more paired historical evidence, while the accuracy benefit remains unestablished. Missing or unrequested pages were not scored as unsupported. No new forecast, agent call or final-target access was needed.

The undeployed 094 unsupported-card compression is not an urgent change for these pilot pages: no displayed pair was wholly unsupported. The 095 historical-neighbor index still needs an actual-current-context audit before any prospective trial; richer general pair support does not establish supported one-setting neighbors or a useful recommendation. Keep the live 097 worker unchanged while measuring accumulated outcomes.
