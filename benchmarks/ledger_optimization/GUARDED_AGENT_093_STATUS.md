# Guarded pilot 093 — live partial audit

Snapshot: 2026-09-14T15:16:50.115118+00:00. 10/36 sessions independently audited; 657 checks, zero integrity failures. Runtime: published Gnomon 1.2.0; model: Engy deepseek-v4.1-flash.

| Arm | Valid forecasts | Full workflows |
|---|---:|---:|
| plain | 3/3 | 2/3 |
| gnomon | 4/4 | 4/4 |
| ledger | 3/3 | 3/3 |

All-three matched subset: 3 tasks, including baseline-only outcomes. Mean per-case RMSLE: plain=0.402479, gnomon=0.366887, ledger=0.361452.

Do not compare unpaired arm means while the run is partial. This pilot checks feasibility and integrity; accuracy does not determine promotion. The 20% held-out target remains unmet.

The first incomplete plain session issued 19 data_summary calls and exhausted 16 model requests, fitting only the seasonal baseline (4 numerical attempts). One wrong column name was corrected. Its valid forecast remains scored; it is not counted as a full ML workflow.

Future UX follow-up: expose exact available columns and a task-preserving correction for data_summary. No running source changes or selective reruns.

Evidence: `evidence/guarded-agent-093-pilot-audit-002.json`; full verified snapshot under `results/guarded-history-093-pilot-audit-002`.

Dispatch is still the frozen 36-session pilot. The gate requires at least 11/12 full workflows in each arm and a complete integrity audit. No automatic full-development dispatch. Final holdout remains closed; main/PyPI unchanged.
