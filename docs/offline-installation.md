# Offline and air-gapped installation

Gnomon's built-in runtime has no required third-party Python dependencies and
does not need an account, API key, model service, or network connection while
running. A controlled deployment can therefore install one reviewed wheel and
serve the CLI, Python API, and MCP interface entirely inside the boundary.

This is an offline-operability claim, not a regulatory certification. Your
organization still owns artifact retention, access control, change approval,
host hardening, and any controls required by a named standard.

## Prepare the transfer on a connected machine

Download the wheel for the exact approved version and record its checksum:

```bash
python -m pip download --only-binary=:all: --no-deps \
  gnomon-forecast==0.7.0 --dest wheelhouse
sha256sum wheelhouse/gnomon_forecast-0.7.0-py3-none-any.whl \
  > wheelhouse/SHA256SUMS
```

Transfer the wheel and `SHA256SUMS` through your normal reviewed media or
artifact-promotion process. Do not transfer a mutable `main` checkout when the
deployment requires a reproducible release.

## Verify and install inside the boundary

Python 3.11 or newer must already be available. No package index is needed:

```bash
cd wheelhouse
sha256sum --check SHA256SUMS
python3 -m venv /opt/gnomon/venv
/opt/gnomon/venv/bin/python -m pip install \
  --no-index --no-deps ./gnomon_forecast-0.7.0-py3-none-any.whl
/opt/gnomon/venv/bin/gnomon --version
/opt/gnomon/venv/bin/gnomon capabilities
/opt/gnomon/venv/bin/gnomon self-check leakage --cases 8 --seed 7
```

Use the absolute executable path in an MCP client so the host does not invoke
an online package resolver:

```json
{
  "mcpServers": {
    "gnomon": {
      "command": "/opt/gnomon/venv/bin/gnomon",
      "args": ["mcp", "serve", "--profile", "core"]
    }
  }
}
```

`gnomon capabilities` reports the installed runtime version, active MCP
profile, optional backends actually present, and the public product-claim
boundary. Retain that output with deployment approval evidence.

## Optional components

The built-in classical runtime is the supported zero-dependency offline path.
Parquet, Excel, StatsForecast, and TSFM adapters are optional:

- Build a reviewed wheelhouse containing every transitive dependency before
  transferring an optional Python extra. Install it with `--no-index` and
  `--find-links`, never by relaxing the boundary temporarily.
- TSFM sandboxes and model weights normally download on installation or first
  inference. They are not available in an air-gapped deployment unless their
  pinned packages and weights have been separately mirrored and validated.
- `uvx --from gnomon-forecast ...`, the Bash URL installer, and direct GitHub
  installs are connected-install conveniences. Do not use them offline.

Missing optional components are reported by `gnomon capabilities`; they do
not make the built-in runtime fail or silently substitute a remote service.

## Upgrade and rollback

Treat each wheel as a separate release. Verify its checksum, install it into a
new virtual environment, run the structural self-check and a representative
local-data forecast, then atomically update the MCP command or executable
symlink. Keep the previous environment until the new artifact and tier
semantics have passed local acceptance checks.

The repository's `scripts/offline_wheel_smoke.py` performs this clean-wheel
journey. Pull-request and release CI run it inside a network-disabled container
so offline operability is a build gate rather than a documentation promise.
