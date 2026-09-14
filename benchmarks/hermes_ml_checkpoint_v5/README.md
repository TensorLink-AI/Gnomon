# Corrected-history agent trial (092)

See PROTOCOL.md for the frozen treatment and gates. Forked from checkpoint-v4;
run the preflight with the pinned runtime before pipeline. No API calls are
made by preflight (native Hermes tests use synthetic upstream responses).

```sh
export LEDGER_ML_RUNTIME_ROOT=/root/gnomon-ledger-ml-v3/runtime
export LEDGER_ML_TASK_SOURCE=/root/gnomon-ledger-ml-v3/code/results/ledger-ml-continuous-022/export-002/host-jobs.json
"$LEDGER_ML_RUNTIME_ROOT/gnomon-venv/bin/python" -m benchmarks.hermes_ml_checkpoint_v5.preflight --output results/corrected-history-092-preflight-001
"$LEDGER_ML_RUNTIME_ROOT/gnomon-venv/bin/python" -m benchmarks.hermes_ml_checkpoint_v5.pipeline --root results/corrected-history-092-run-001 --preflight results/corrected-history-092-preflight-001/passed.json
```

Use new output paths for each separately documented attempt; preserve failures.
The pipeline checks original v3 setup assets/task hashes and protected source
hashes before dispatch. All three arms start with fresh memories and ledgers.
