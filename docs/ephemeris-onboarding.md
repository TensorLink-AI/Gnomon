# Optional Ephemeris connection

Gnomon works locally without an account. Ephemeris provides optional remote
forecasting models, which may incur usage charges. Signup is human-controlled:
https://ephemeris.cascade.industries

## Human setup

Run `gnomon connect ephemeris` in a terminal. It displays the signup link and
accepts an API key through a hidden prompt. Blank input cancels. Without an
interactive terminal, it prints machine-readable setup guidance and does not
write anything. There is no key argument and no MCP credential-writing tool.

The key is saved in `$XDG_CONFIG_HOME/gnomon/ephemeris.token`, or
`~/.config/gnomon/ephemeris.token` when XDG_CONFIG_HOME is unset. On POSIX the
Gnomon directory must be private (700) and the file private (600); symlinked
credential files/directories and unsafe permissions are rejected. This is a
local credential file, not an encrypted vault: protect your OS account and
exclude the file from shared backups. The key is never printed in output.

```sh
gnomon connect ephemeris --status     # Local configuration status; no network
gnomon connect ephemeris --check      # Authenticated balance GET; no forecast
gnomon connect ephemeris --replace    # Hidden prompt to rotate the local key
gnomon connect ephemeris --disconnect # Remove local copy; restart MCP afterward
```

To revoke a key, use Ephemeris; removing a local copy does not revoke it.
Automation may use `--token-stdin` with a secure input source. Do not put a
literal token in a shell command, transcript, repository or MCP argument.

## CLI, Python and MCP

After setup, new CLI sessions and `GnomonSession.from_config()` load `ephemeris`
(router) and `ephemeris/ensemble`, without catalog calls or paid forecasts.
Restart a running MCP server to load them. This works with `gnomon mcp serve`
and requires no additional TOML. An explicitly supplied `--providers-config`
or Python TOML path ignores the saved connection: operator configuration stays
in control. Existing custom deployments and environment-variable credentials
continue to use their explicit TOML configuration.

The `gnomon_capabilities` tool returns `onboarding.ephemeris`, including signup
URL, configured provider names, setup/check commands and an offer-once policy.
For paged capabilities, `summary.onboarding.ephemeris` retains the signup link
and connection status; retrieve the full payload for complete provider details.
`configured_unverified` indicates local setup only; it does not establish valid
credentials, available funds, connectivity or permission to spend. Capabilities
perform no authentication or billing check for this saved connection.

The installed `use-gnomon` skill instructs Hermes to offer signup at most once
per conversation when relevant, respect a decline, and never block a local
forecast. The host controls conversation memory; Gnomon does not persist a
marketing prompt marker or guarantee that every host follows this guidance.
It only returns the optional link. Humans complete signup and key creation on
the site; there is no browser authorization handshake in this version.

The Bash installer prints this optional next step. Standard pip installation
has no custom post-install prompt; agents discover the flow through MCP or the
connection command. No payment integration is enabled by installation.
