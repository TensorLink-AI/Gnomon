# Completed Hermes ML comparison and native-memory follow-up

Gnomon **1.2.0**, DeepSeek **deepseek-v4.1-flash** through Engy. Four reused
Favorita retail series, 26 sequential origins each, 14-day horizon. Agents
backtested and iterated seasonal, Ridge and random-forest configurations under
the same numerical and API limits. These are development results, not final
held-out evidence or a broad time-series task suite.

| Arm | Mean case RMSLE | Completed workflows | Reported tokens | API calls |
|---|---:|---:|---:|---:|
| Hermes | 0.480148 | 99/104 | 25,723,263 | 1,368 |
| Hermes + Gnomon | 0.481454 | 102/104 | 24,581,379 | 1,380 |
| Hermes + Gnomon + ledger | 0.475970 | 102/104 | 29,344,909 | 1,364 |
| Hermes + explicitly prompted native memory | 0.475949 | 104/104 | 26,831,582 | 1,354 |

All 416 sessions produced valid forecasts. Workflow completion additionally
requires comparison and selection after backtesting; a valid initial forecast
alone is insufficient. The memory arm was a later follow-up; only that arm was
rerun. This is not a simultaneous randomized four-arm experiment.

Ledger reduced RMSLE by **1.14%** versus no-ledger Gnomon. The exploratory
four-series bootstrap 95% interval is **-2.13% to +4.20%**. Mature-history
ledger RMSLE was 0.489413 versus 0.482836 without ledger. Neither the overall
nor mature result establishes accumulated-history benefit; the 20% target
remains unmet. Native memory and ledger have effectively identical error.
Native memory used 8.6% fewer reported tokens than ledger, but main ledger
and plain usage records are incomplete and billing costs were unavailable.

Native memory was exercised: 53 sessions invoked the memory tool, 99 invoked
skill reading and 92 invoked skill updates. These are tool invocation counts,
not evidence that every saved lesson was correct or useful.

## Audit and preserved failures

The original main audit passed **59,003 checks**. The memory audit initially
stopped on a missing shutdown record after a worker was terminated with -15.
That session had already saved a valid, compared forecast. The revised read-only
audit checks the recorded termination, elapsed budget, raw request history and
unchanged checkpoint, and reports the missing shutdown record separately.
It passed **19,312 checks**, with no numerical/integrity failures and one
explicit shutdown-record gap. Seven regression tests cover the repair and
historical visibility checks. No model call or rescoring mutation was performed.

Original BLOCKED/INCOMPLETE records remain intact. Both original archive hashes
match their terminal receipts. All 460 selected result/grade/job files match
the original inventories. Re-audit output is separate under
`results/ml-completed-030-reaudit/`; original scores and forecasts are unchanged.
Exact results, costs, hashes and limitations are in `evidence/ml-completed-030.json`.

## What the opportunity check changes

The best already-executed forecast in hindsight would reduce the ledger arm's
RMSLE only from 0.475970 to 0.465477: **2.20%**. There were better executed
alternatives in 12/104 tasks. This does not bound models never executed and
does not establish a deployable policy.

Although 101 tasks executed at least two configurations, only 16 tasks had
three or more historical origins matched across every current configuration.
The changing configuration portfolio often lacks directly comparable accumulated
history. Exact per-task coverage and immutable input hashes are preserved in
`evidence/ml-completed-opportunity-031.json`.

The next infrastructure experiment should test a persistent comparison cohort:
retain stable, versioned incumbent configurations across origins, record the
same completed horizons for challengers, and expose evidence sufficiency before
requesting a historical rank. Give all arms the same configuration opportunities,
raw information and numerical budget. Freeze the cohort rule before another
development evaluation; do not select a portfolio using future outcomes or
reinterpret this run as evidence of a 20% gain. Main, PyPI and the untouched
final evaluation remain unchanged.
