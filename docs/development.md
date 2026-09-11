# Development

```bash
python -m pip install -e '.[dev]'
ruff check src/gnomon
pytest -q tests benchmarks/tests
python -m compileall -q src tests
python scripts/check_production_progress.py
```

The runtime has one execution path: session joins Python/CLI/MCP; inference and
forecast_adapter define provider execution; ephemeris is an optional HTTP connector.
Data/data_refs/datasets/repair/temporal/temporal_store own input and snapshot semantics.
Ledger owns optional durable evidence and rejects incompatible database schemas.
Backtesting and study_routing consume that evidence; temporal_ops is opt-in.

Project dependencies are declared in `pyproject.toml`. The ordinary benchmark environment has its own hash-locked
`benchmarks/workflow/software/requirements.txt`, separate from the core package.
Per-library adapters live in `gnomon.adapters`, one module per `kind`, each behind a
pip extra with a lazy import and a skip-if-missing test. Add one only for a maintained,
PyPI-released package; everything else stays a user-owned callable/factory integration.
Unrelated user data and scratch files are not part of product changes.

Production tests cover retained semantics. `benchmarks/workflow` measures the
current product with an ordinary, lean and optional-features-enabled arm; its
scoring and model-client code do not ship in the wheel. No tests authorize paid
model calls. Container checks require explicit locally built immutable image IDs.

Build a wheel and run `scripts/offline_wheel_smoke.py` outside the checkout.
Rebuild the benchmark service image whenever package bytes change; its source
fingerprint rejects stale installed code. [Release process](ci-cd.md).
