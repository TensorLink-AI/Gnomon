# Development

```bash
python -m pip install -e '.[dev]'
ruff check src/gnomon
pytest -q tests
pytest -q benchmarks/tests
python -m compileall -q src tests
python scripts/check_production_progress.py
```

Use Python 3.11–3.13. Production regressions and current evaluation-harness tests
are separate gates. Neither authorizes paid model requests. Container tests skip
without operator-supplied image IDs; see [benchmarks](../benchmarks/README.md).

## Layout

- `src/gnomon`: shipped toolkit. `session.py` joins Python/CLI/MCP;
  `inference.py` and `forecast_adapter.py` define provider execution;
  `ephemeris.py` handles the service; `ledger.py` preserves evidence.
- `tests`: production regressions, including retained legacy runtime compatibility.
- `examples/provider_plugin`: independently installable provider walkthrough.
- `benchmarks/workflow`: current matched evaluation, not a runtime dependency.
- `docs`: current usage, reference and operations.
- `docs/production`: detailed contracts and the scoped delivery checkpoint.

Removed benchmarks, duplicate source archives and design proposals are recoverable
from checkpoint commit `2cba20e`. Use Git history instead of restoring competing
docs or importing retired benchmark modules into the runtime.

## Legacy implementation boundaries

The default execution session does not load the legacy registry. For retained
advanced profiles, use these owners instead of adding more logic to `toolspec.py`:

| Module | Responsibility |
| --- | --- |
| `tool_catalog.py`, `tool_profiles.py` | Schema and profile metadata; no runner imports |
| `tool_input.py`, `tool_context.py` | Schema/horizon resolution and context/covariate preparation |
| `tool_operations.py` | Execute retained legacy operations |
| `tool_publication.py` | Project and record publication evidence without changing forecasts |
| `tool_response.py` | CLI/MCP response formatting; no registry or runtime imports |
| `toolspec.py` | Legacy MCP dispatch and compatibility exports |

Runtime capability reporting consumes metadata, not dispatch. CLI formatting imports
response functions directly. The unregistered experimental dispatchers are removed;
their underlying decision and tracking APIs remain tested directly. Compatibility
tests pin legacy schemas/profiles and verify these import boundaries in fresh processes.

## Safe changes

Validate request/result semantics and add regressions for malformed data,
unsupported capabilities, leakage and failure accounting. Keep fit-per-request
models fresh during evaluation. Successful execution is not evidence of provider
accuracy, calibrated uncertainty or action authority.

Build with `python -m build`, then run
`python scripts/offline_wheel_smoke.py dist/*.whl`. The smoke script accepts
`--example-wheel` for the separate plugin package. Rebuild the benchmark service
image whenever package bytes change: its fingerprint rejects stale installed code.

See [releases](ci-cd.md) and [validation limits](agent-evaluation.md).
