# Checkpoint-first v3 Hermes ML workflow

This is a separate experiment from hermes_ml_iteration. The original96-session
results remain unchanged. See PROTOCOL.md for the preregistered pilot gate and
conditional evaluation. The numerical models and Gnomon1.2.0 build stay fixed.

Run from the repository with the exact matching environments under OTHER in run.py:

```sh
env -u PYTHONPATH /path/to/gnomon-venv/bin/python \
  -m benchmarks.hermes_ml_checkpoint_v3.preflight --output /new/root/preflight-001
env -u PYTHONPATH /path/to/gnomon-venv/bin/python \
  -m benchmarks.hermes_ml_checkpoint_v3.pipeline --root /new/root
```

The pipeline runs36 pilot sessions, audits them, and writes GATE.json. Only if EACH
arm completes >=11/12 full workflows does it launch a new96-session evaluation,
using fresh homes/ledgers. Completed sessions are never rerun. Within a session, up to two explicit
corrections for premature text/repetition share its original request/time budget. A failed gate
is a completed pilot result, not an infrastructure failure. BLOCKED.json records
infrastructure/audit failure. FINISHED.json and the checksummed evidence archive
are written after the last permitted phase.

`lab.py start` saves a tested baseline. `backtest --config` admits three folds under
one lock while retaining a final-fit reserve. `commit --config` explicitly selects
a tested configuration; duplicate existing executions require --execution-id.
The authoritative checkpoint.json is atomically replaced only after validation,
with immutable checkpoint copies and a selection log. Final chat is not graded.
Results/errors expose budgets. Baseline survival and full workflow completion are
reported separately; comparison without a subsequent explicit selection is incomplete.

preflight.py tests exact cross-backend predictions, baseline-only classification,
concurrent budget admission, exhausted-budget final commit, failed replacement,
ambiguous selections, origin reset and outcome maturity. The independent report
audit recomputes metrics and verifies checkpoint selection against evidence that
existed at the selection event. Original evidence/source hashes are retained.

For reproduction, restore the pinned Hermes archive,1.2.0 wheel and matching package
inventories from setup, and original host-jobs to setup/recovered-task-source under
the restored OTHER root. Supply the Engy key only to the host configuration and use
new output directories. Never rerun mutation helpers on archived evidence. This is
a development workflow test on previously used series, not a held-out confirmation.

V2 reserves requests13–16 and the final90 seconds for selection. It suppresses
Hermes’ extra post-budget summary request locally. See policy.py and PROTOCOL.md.
The offline test_orchestration module uses real pinned Hermes/native tools with
synthetic upstream responses to check correction, the final-call commit, budget
enforcement, and phase restrictions without spending on model inference.


V3 adds common outcome maturation for all matching prior production executions,
including unselected alternatives, in all three arms. It preserves the stopped
v2 pilot and runs separately on the Targon pod. See PROTOCOL.md. Runtime paths
are supplied via LEDGER_ML_RUNTIME_ROOT; restore_runtime.py rebuilds the exact
archived package inventories in a new durable directory before preflight.
