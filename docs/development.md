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
Ledger and artifact_import own durable evidence and read-only migration.
Backtesting and study_routing consume that evidence; temporal_ops is opt-in.

There is no retained legacy runtime, registry, publication stack or model installer.
Project dependencies are declared in `pyproject.toml`; the stale root `uv.lock`
was removed. The ordinary benchmark environment has its own hash-locked
`benchmarks/workflow/software/requirements.txt`, separate from the core package.
Do not add per-library adapters: users own callable/factory integrations.
Git checkpoint `333ed2c` preserves the removed implementation and its tests.
Unrelated user data and scratch files are not part of the cull.

Production tests cover retained semantics. `benchmarks/workflow` measures the
current product with an ordinary, lean and optional-features-enabled arm; its
scoring and model-client code do not ship in the wheel. No tests authorize paid
model calls. Container checks require explicit locally built immutable image IDs.

Build a wheel and run `scripts/offline_wheel_smoke.py` outside the checkout.
Rebuild the benchmark service image whenever package bytes change; its source
fingerprint rejects stale installed code. [Release process](ci-cd.md).
