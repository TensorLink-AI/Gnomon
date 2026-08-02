# Development and testing

## Repository layout

```text
src/aion/       Runtime and CLI source
tests/               Unit and end-to-end tests
docs/                User and developer documentation
examples/            Small runnable datasets
.github/workflows/   CI, PyPI release, and container automation
Dockerfile           Production CLI container image
install.sh           Isolated one-command installer
pyproject.toml       Package metadata and test configuration
```

## Run from source

```bash
cd Aion
PYTHONPATH=src python3 -m aion capabilities
```

## Run tests

```bash
cd Aion
PYTHONPATH=src pytest -q
```

The tests cover schema inspection, model selection on a known trend, artifact
persistence, unsupported short series, duplicate timestamp errors, capability
truthfulness, and CLI structured errors.

## Build distributable packages

```bash
uv build
```

This creates a source distribution and wheel under `dist/`. To verify the wheel
without changing your normal environment:

```bash
uv venv /tmp/aion-wheel-verify
uv pip install \
  --python /tmp/aion-wheel-verify/bin/python \
  dist/aion_forecast-0.1.0-py3-none-any.whl
/tmp/aion-wheel-verify/bin/aion capabilities
```

## Design constraints for contributions

- The numerical runtime owns every number and support decision.
- All candidate methods compete against mandatory baselines.
- Evaluation operations must preserve temporal order.
- Public capability discovery must reflect tested functionality.
- Unsupported analysis is distinct from invalid input.
- Source data is read-only; outputs go into new run directories.
- Avoid adding a documented command before its end-to-end path works.

See [design review decisions](../DESIGN_REVIEW_NOTES.md), the broader
[product specification](../Aion_MVP_Product_Specification.md), and
[system design](../Aion_System_Design.md).

Release maintainers should also read [CI/CD and release operations](ci-cd.md)
and [Containers](containers.md).
