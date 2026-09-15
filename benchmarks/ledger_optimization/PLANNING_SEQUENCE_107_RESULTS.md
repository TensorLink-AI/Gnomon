# Checkpoint-first recipe sequence 107 — structural and synthetic pass

The explicit prerequisite fixes the immediate-action dead end measured in 106.
On the same 63 recorded reviews, 46 now have a budget-feasible first `start`
action and conditional historical-recipe backtests. This is not a forecast
accuracy result or proof that an agent will use the plan. No paid arm is changed
or launched by this experiment.

## Recorded-context replay

Protocol freeze `7fe5d1f2` preceded implementation/replay; implementation was
committed as `80a21c57` before real-context replay. All 63 reviews remain included.
Four cold reviews have no historical catalog. Of the 59 rendered reviews, 58
lacked a checkpoint; the one with a checkpoint had no eligible untried recipe.

The planner excludes seasonal-7 from later recipes because the required `start`
step will itself execute it. That is a prospective dependency, not a fabricated
execution record. It returns 91 supported recipe entries across the 46 feasible
sequences. Every dependent backtest remains `admissible_now: false` until the
checkpoint exists. Original observed state, source references and shared budget
remain explicit. These are repeated entries, not 91 independent opportunities.

The replay passed 1,082 checks, including authenticated original review replies,
support unions, recipe identity/order, baseline exclusion, next-call arguments,
interaction/numerical capacity and unchanged inputs. It made no fits, ledger
queries or Engy requests. Eighteen unit tests passed across 106 and 107.

## Executed synthetic sequence

A separate scripted probe used the unmodified common lab and the published
Gnomon 1.2.0 runtime. Four synthetic origins accumulated matched seasonal-7 and
Ridge production outcomes. At the fifth origin it executed:

1. Read the actual ledger review and construct the conditional plan.
2. Execute the plan's common `start` prerequisite: four numerical attempts.
3. Re-render against the real checkpoint and newly returned budget.
4. Backtest the proposed exact Ridge configuration: four additional attempts.
5. Explicitly commit its executed forecast: no additional numerical attempt.

The selected execution preserved series, unit, origin and future timestamps,
with two distinct complete three-fold backtests including Ridge. All earlier
event-log prefixes and frozen source files were preserved. The probe passed
175 checks and recomputed each CV RMSLE from saved points/actuals.

Across setup and the tested sequence there were five synthetic origins, 40
numerical attempts, 40 successful results, 20 Ridge estimator fits and 21
scripted tool commands. No Engy call was made. Scripted tool-step budget counters
are not actual agent/API requests. The tested final sequence cost eight numerical
attempts, within the explicitly projected bound and with its reserve intact.
This tests one recipe and a small local flow, not arbitrary tool failures or
real agent behavior.

An additional read-only check verified all ten configuration/origin groups have
three CV results and one production result, with no missing attempts. It also
queried the isolated installed environment: build
`1.2.0+ga38cd0cad353.s9723394ccb6d`, source fingerprint
`9723394ccb6d9e11991b312e01bac47c767c69407b6b33d36971cb6e48b6a22e`.

The first post-check mistakenly resolved the venv interpreter symlink to the
system Python, which had no Gnomon installed. Its command/output/failure is
retained in `planning-sequence-107-postcheck-001`. The corrected check preserves
the venv invocation path. No synthetic forecast was rerun for that correction.

## Evidence and next gate

The archive contains 472 inventoried files plus its inventory, with every member
streamed and hash-checked after writing. Compressed size: 4,375,044 bytes;
SHA-256 `a11c4a3d7f8839b30a3aa1b803a1c076b07793f45edb12107f40ed62b4666ff3`.
Original files remained unchanged during archival.

- [Compact evidence receipt](evidence/planning-sequence-107-result-001.json).
- Recorded-context replay: `results/planning-sequence-107-offline-001/`.
- Synthetic projects, commands and outcomes: `results/planning-sequence-107-synthetic-001/`.
- Launch outputs: `results/planning-sequence-107-launch-001/` and `planning-sequence-107-synthetic-launch-001/`.
- Post-checks: `results/planning-sequence-107-postcheck-001/` and `-002/`.
- Archive: `results/planning-sequence-107-archive-001/evidence.tar.gz`.

Before a paid agent comparison, integrate the plan at the actual review boundary,
freeze its exact presentation and sources, and validate the resulting worker
with the same control tools, evidence access and budgets. Current candidate-100
and M5 runs remain unchanged. The 20% target and uncertainty requirement are
unmet; no held-out values, main branch or PyPI release were touched.
