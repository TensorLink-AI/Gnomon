# Hermes ML checkpoint v4

Three arms: Hermes, Hermes + Gnomon 1.2.0, Hermes + Gnomon 1.2.0 + development ledger cards.
See PROTOCOL.md for the frozen design, budgets, service admission and limitations.

Set LEDGER_ML_RUNTIME_ROOT to the sealed runtime and LEDGER_ML_TASK_SOURCE to the
continuous host-jobs.json. Run preflight with the Gnomon runtime interpreter:

```sh
python -m benchmarks.hermes_ml_checkpoint_v4.preflight --output /path/to/new-preflight
python -m benchmarks.hermes_ml_checkpoint_v4.pipeline --root /path/to/new-run --preflight /path/to/new-preflight/passed.json
```

The 36-session completion pilot gates a fresh 312-session development evaluation.
No accuracy-based promotion, no automatic retry of failed sessions, no held-out
claim, and no changes to published Gnomon. Every run uses a new directory.
