# Guarded Hermes development evaluation 093 — complete sessions

All 312 scheduled sessions completed, covering four reused Favorita retail
series, 26 sequential origins and three arms. Published Gnomon **1.2.0** and
Engy **deepseek-v4.1-flash**, requested seed 7, were used throughout. The
development continuation retained the 36 pilot sessions exactly once.

The ledger arm completed more workflows and used fewer reported tokens, but
this run does **not** establish the 20% accuracy objective or an increasing
accuracy benefit from accumulated history.

| Arm | Mean per-case RMSLE ↓ | Valid forecasts | Full workflows | Reported tokens | API requests | Numerical fits |
|---|---:|---:|---:|---:|---:|---:|
| Hermes | 0.48468771 | 104/104 | 97/104 | 29,796,634 | 1,455 | 1,457 |
| Hermes + Gnomon | 0.48461626 | 104/104 | 99/104 | 28,588,617 | 1,403 | 1,533 |
| Hermes + Gnomon + development ledger | 0.47199148 | 104/104 | 104/104 | 23,103,796 | 1,220 | 1,683 |

Ledger reduces RMSLE by **2.61%** versus Gnomon without ledger and **2.62%**
versus plain Hermes. It uses **19.19% fewer reported tokens** and **13.04% fewer
API requests** than Gnomon without ledger, but **9.78% more numerical fits**.
These quantities describe different resources; token savings are not a claim
about total dollar cost or uniformly lower computation.

## Accuracy uncertainty and accumulation

The predeclared exploratory series bootstrap uses 2,000 draws with seed 142.
The ledger-versus-Gnomon relative-improvement interval is **−2.01% to +6.42%**;
it crosses zero. Ledger wins 52 matched cases, loses 36 and ties 16. The
ledger-versus-plain interval is +0.71% to +3.81%, but four reused series and one
requested seed do not provide confirmatory evidence for general superiority.
No undefined zero-control draws occurred in the complete cohort.

| Predeclared phase | Matched cases | Gnomon RMSLE | Ledger RMSLE | Ledger error reduction |
|---|---:|---:|---:|---:|
| Cold: origins 0–3 | 16 | 0.520591 | 0.491140 | +5.66% |
| Mature: origins 10–25 | 64 | 0.484294 | 0.484394 | −0.02% |
| Later: origins 22–25 | 16 | 0.547291 | 0.563311 | −2.93% |

The mature and later windows overlap. Their different calendar periods and
task difficulty prevent a causal interpretation of changes across phases.
Nevertheless, this run supplies no increasing mature-history accuracy gain:
ledger is effectively tied in the mature window and worse in the last four
origins. It received numerical ledger evidence in 100/104 sessions, with up to
25 earlier origins; 15 sessions received recent-versus-lifetime disagreement.
Evidence availability does not demonstrate correct reliance on it.

On the 92 cases where all three arms completed the full workflow, RMSLE is
0.487338 / 0.489950 / 0.478688. This successful-only subset is a diagnostic,
not the primary estimate: selecting cases by completion can bias comparison.

## Completion, fairness and costs

All arms had the same raw information, model/configuration space, time and
request budgets, numerical limits, and native-memory availability. The ledger
treatment includes the frozen comparison and concise evidence helper; it does
not isolate database storage from evidence presentation. Native memory was
saved in 36 plain-Hermes sessions, with six native memory/skill tool calls;
the other arms made two and one such calls respectively. Saved-session counts
include repeated exposure to retained notes, not distinct writes.

Full completion requires at least two distinct configurations backtested on
three current folds, at least one ML model, and an explicit selection after
comparison. All twelve incomplete workflows returned valid seasonal baseline
forecasts, exhausted the 16-request budget, and performed only four numerical
fits. All remain scored. The last plain-Hermes session attempted to commit a
Ridge configuration without backtesting it on the current folds; rejection
preserved the baseline. A decision summary alone did not satisfy the workflow.

Reconciled forecast usage is **4,078 requests and responses**, **81,489,047
reported tokens**, **zero API errors**, zero missing forecast usage and zero
orphan responses. Readiness probing added 315 requests and 4,368 known tokens;
three timed-out probes have unknown usage and remain disclosed. Dollar billing
was not supplied. No sessions were rerun and no failures were removed.

## Evidence and next step

Disjoint local audits cover all 312 sessions with **87,285 checks**, zero
integrity failures and no missing shutdown records. Every transferred file was
hash checked, retained pilot prefixes were verified, and all 312 rows agree
exactly with the remote full audit. The experiment process exited 0. Controller
archive packaging is still pending at this report revision; the queued
synthetic seed validation has not been accepted as passed.

Authoritative receipts:

- `evidence/guarded-agent-093-development-final-001.json`: complete scores,
  exploratory uncertainty, phases, per-series results, all incomplete rows,
  source hashes and reconciled costs.
- `evidence/guarded-agent-093-development-audit-024.json`: the final ten sessions
  and the complete disjoint audit assembly.
- `evidence/guarded-agent-093-failure-monitor-006.json` and `007.json`: final
  incomplete-session diagnostics and retained raw-evidence references.

The prepared 096 experiment changes prospective collection equally in all
arms: each backtested configuration also produces a metered current forecast,
so more comparable outcomes can mature later. It uses fresh controls and state;
093 outcomes are not reused as 096 controls. This tests a measured evidence
coverage gap rather than assuming that longer history alone will improve
selection. It remains unlaunched until the controller and queued validation
finish successfully. The untouched final holdout remains closed, the **20%
objective is unmet**, and main/PyPI are unchanged.
