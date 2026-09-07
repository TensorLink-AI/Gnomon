# Offline installation

Build a reviewed wheel on a connected machine:

```bash
python -m pip install build
python -m build --wheel --outdir wheelhouse
sha256sum wheelhouse/*.whl
```

Record its commit and SHA-256, transfer it through your normal approval process,
verify the hash, then install in the target Python 3.11–3.13 environment:

```bash
python -m pip install --no-index --no-deps wheelhouse/gnomon_forecast-1.0.0-py3-none-any.whl
gnomon capabilities
gnomon self-check leakage --cases 8
```

For repeatable deployment, retain the exact wheel and checksum.

The core and three reference baselines need no network service. Your chosen
model software, weights and optional Parquet/Excel readers must be provisioned
separately with their full dependency set. Gnomon does not download or manage them.

Configure the agent host with the installed executable's absolute path and
`mcp serve`. Add explicit operator TOML for local providers and a ledger.
Do not use an online resolver in an offline host.

`scripts/offline_wheel_smoke.py` validates an installed wheel outside the checkout.
CI also runs it in a network-disabled container with a separately built
[user-provider example](../examples/provider_plugin/README.md).
