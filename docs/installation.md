# Installation

Use Python 3.11–3.13. The distribution is `gnomon-forecast`; the Python import and
CLI command are `gnomon`. The core requires no third-party packages.

## Release candidate

```bash
python -m pip install --pre 'gnomon-forecast==0.8.0rc2'
gnomon --version
```

A prerelease is explicitly selected: plain `pip install gnomon-forecast` may
continue to choose the previous stable release. Before this candidate is published,
install the reviewed checkout instead.

## Checkout

```bash
python -m pip install -e '.[dev]'
# Or install the checkout into the installer's isolated environment:
bash install.sh --local
```

Without `--local`, the Bash installer fetches the configured repository/ref,
not your working changes. It creates an isolated environment and retains older
installs. Use `bash install.sh --help` for paths and version pinning.
Do not execute a mutable remote installer without reviewing/trusting its source.

## Isolated command

```bash
uv tool install 'gnomon-forecast==0.8.0rc2'
# From the checkout:
uv tool install .
```

Optional readers use the `parquet` or `excel` extras. Your own forecasting library
belongs in the environment containing your callable/factory. Gnomon's legacy
`statsforecast` extra is separate from the generic provider boundary.

For restricted environments, use [offline wheels](offline-installation.md).
Continue with [first run](getting-started.md), [MCP](quickstart-mcp.md), or the
[installable provider walkthrough](../examples/provider_plugin/README.md).
