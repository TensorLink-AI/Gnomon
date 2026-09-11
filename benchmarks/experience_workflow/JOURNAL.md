# Evaluation development journal

## 001 — initial harness (c2119a5), 2026-09-11

Protocol and harness committed before live calls. Independent scorer, SQLite and
public Gnomon matched on 864 full-length checks across four synthetic worlds.
Deliberate visibility/unit/context/revision/no-memory faults were detected. Eight
regression tests passed. Historical-query answers survived later ingestion.

The initial live pilot was interrupted for harness improvements, not retained as
a treatment comparison. Both arms sometimes submitted one flat evidence answer;
the shared submission schema only said `object` and did not explicitly describe
the required original/current children. Also the two vintage queries used
different origin-window endpoints, which confounded evidence revision with newer
forecast origins. Both issues belong to the benchmark, not Gnomon forecasting.

Retained original run: `results/experience-workflow/pilot-001/`, with source
manifest, wire logs and five completed checkpoints. 52 requests were started;
48 responses retained; 182,404 billed tokens are known. Four interrupted requests
have unknown response usage; total billed tokens and dollar cost are unknown.
The process exited 143 after targeted termination. All unsuccessful/incomplete
attempts remain in the evidence. No performance claim is made from this pilot.

## 002 — revision before second live pilot

Fully specify the shared nested submission schema and the Gnomon query shape;
state that empty cold-start cohorts are expected. Original/current now share one
origin range ending eight days before the checkpoint; only evidence cutoffs
change. Persist notes/query registry/event receipts after mutations and add a
cooperative STOP sentinel. Keep model, budgets, primary target, control reference
query, fixed selection rule and demand generator unchanged.

Pilot 002 is another development harness check, not a confirmation or a tuning
success. Record its outcome even if SQLite is cheaper or Gnomon misses a gate.

## 003 — feature and control audit before another pilot

Pilot 002 was stopped cooperatively after a further validity review. Its
forecast histories used final historical measurements even when corrections
were not yet visible. Equal exposure across arms is insufficient for a temporal
safety claim. Replace those features with both-clock-visible vintages and
explicit causal forward fills; audit every generated feature request and test
that changing unavailable revisions cannot change earlier features.

The control also lost its table-schema instructions on chat reset and was denied
read-only table_info requests. Restore schema/saved-query discovery every round,
allow that read-only pragma, and format the reference SQL's existing scores into
the same compact answer as Gnomon. These strengthen the control. Retain pilot
002 but make no treatment-effect claim from these known harness limitations.

## 004 — complete development pilot and metric-disclosure correction

Pilot 003 completed all 64 decisions, with unchanged source and no API errors:
Gnomon 20/32 correct, SQLite 26/32. Tokens per correct checkpoint were 31,427.95
versus 28,471.08 (ratio 1.10386); no objective gate was established. All 12 failed
Gnomon checkpoints contained a query using the public default MAE, while the task
required RMSLE. The benchmark's compact adapter had removed Gnomon's metric
label, and its schema failed to describe that default accurately. These are
known adapter limitations, so this is retained as harness-development evidence,
not a fair estimate of the product's advantage or disadvantage.

Restore effective metric/query metadata, publish the actual MAE default in the
query schema, and give specific metric-mismatch recovery. Preserve original
scores. Add a regression replaying that failure and its task-preserving correction.
Also implement the guarded confirmation freeze, complete-grid checks and a
one-use reserved-cohort registry. No validation or confirmation worlds have been
opened. The next development run must exercise this corrected source.
