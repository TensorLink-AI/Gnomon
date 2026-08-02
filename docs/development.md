# Development and testing

## Repository layout

```text
src/gnomon/            Runtime, CLI, and MCP server source
tests/               Unit, end-to-end, and golden tests
tests/goldens/       Byte-pinned artifacts; refresh with --update-goldens
docs/                User and developer documentation
examples/            Small runnable datasets, including deliberately messy ones
benchmarks/          Runnable adapters for published external benchmarks
integrations/hermes/ Packaged Hermes Agent plugin
skills/              Agent-facing safe-use skills
.github/workflows/   CI, PyPI release, and container automation
Dockerfile           Production CLI container image
install.sh           Isolated one-command installer
pyproject.toml       Package metadata and test configuration
```

## Run from source

```bash
cd Gnomon
PYTHONPATH=src python3 -m gnomon capabilities
```

## Run tests

```bash
cd Gnomon
PYTHONPATH=src pytest -q
```

The tests cover schema inspection, model selection on a known trend, artifact
persistence, unsupported short series, duplicate timestamp errors, capability
truthfulness, and CLI structured errors. Three suites are worth knowing about
before you change behaviour:

- **Goldens** (`tests/goldens/`) pin whole artifacts byte-exact under a fixed
  clock. A diff there means a published number moved. If the move is
  intended, refresh with `PYTHONPATH=src pytest --update-goldens` and read
  the diff before committing it.
- **Leakage lint** (`tests/test_leakage_lint.py`) is an AST check over
  `GUARDED_MODULES`. It fails on code that reads observations outside a
  snapshot, which is how the structural leakage guarantee stays structural.
- **Doc drift** (`tests/test_docs_current.py`) asserts that the counts and
  command lists in `README.md` and `docs/` still match the shipped surface.

## Build distributable packages

```bash
uv build
```

This creates a source distribution and wheel under `dist/`. To verify the wheel
without changing your normal environment:

```bash
uv venv /tmp/gnomon-wheel-verify
uv pip install \
  --python /tmp/gnomon-wheel-verify/bin/python \
  dist/gnomon_forecast-*-py3-none-any.whl
/tmp/gnomon-wheel-verify/bin/gnomon capabilities
```

## Design constraints for contributions

- The numerical runtime owns every number and support decision.
- All candidate methods compete against mandatory baselines.
- Evaluation operations must preserve temporal order.
- Public capability discovery must reflect tested functionality.
- Unsupported analysis is distinct from invalid input.
- Source data is read-only; outputs go into new run directories.
- Avoid adding a documented command before its end-to-end path works.

For the reasoning behind these constraints, read
[Concepts](concepts.md) — it documents the partitioning, the
baseline rule, and why abstention is a result rather than an error.

The [product specification](../Gnomon_MVP_Product_Specification.md) and
[system design](../Gnomon_System_Design.md) are v0.1 direction documents and
describe features that were never built; check `gnomon capabilities` before
relying on either.

Release maintainers should also read [CI/CD and release operations](ci-cd.md)
and [Containers](containers.md).
