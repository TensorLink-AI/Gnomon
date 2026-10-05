# Connect Gnomon

Inspect time series, forecast with your models, compare forecasts, and review observed outcomes.

Use [installed.json](installed.json) with an existing Gnomon installation. Set
`command` to its absolute executable path if your host does not inherit PATH.
It starts `gnomon mcp serve`: three offline baselines plus any optional saved
Ephemeris connection, with no ledger. Without a saved connection, no credentials
are needed. Use an explicit TOML configuration for a fixed provider set.
This is also the checkout recipe after `python -m pip install .`.

[pypi.json](pypi.json) uses `uvx --from gnomon-forecast==VERSION gnomon mcp serve`.
It installs the pinned released package and requires network access on first use.
It does not run the checkout or guarantee that unpublished features are available.
Use the installed recipe for your local model environment; an isolated uvx
installation cannot import model libraries from another environment.

For a ledger or remote providers, add `--providers-config` and the absolute
operator TOML path to the selected recipe's args. Supply credentials through the
host environment when that provider requires them. Do not put secrets into these
shared files. Neither recipe configures a hosted HTTP MCP endpoint.

After connecting, discover `tools/list`, then call `gnomon_capabilities` with
`{"task":"forecast"}` on builds that support task discovery. Older releases
can use `gnomon_capabilities` with `{}` and the exposed tool schemas.
Follow the [MCP quickstart](../../docs/quickstart-mcp.md) and
[main agent skill](../../skills/use-gnomon/SKILL.md).

The root [server.json](../../server.json) is prepared for the official MCP Registry.
It launches the package-name `gnomon-forecast` alias with `mcp serve`; that alias
and the README ownership marker must be in the published package before this
listing is usable. See [publication checks](../../docs/discovery-and-listings.md).
