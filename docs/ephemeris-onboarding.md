# Optional Ephemeris connection

Gnomon works locally without an account. Ephemeris provides optional remote
forecasting models, which may incur usage charges. Signup is human-controlled:
https://ephemeris.cascade.industries

## Native Hermes setup

After installing Gnomon, the agent can install the two packaged Hermes skills:

```sh
gnomon connect ephemeris --install-hermes-skill
```

This writes `use-gnomon` and `connect-ephemeris` under
`$HERMES_HOME/skills` (default `~/.hermes/skills`). It never reads a key or
changes MCP configuration, and refuses to overwrite different existing skills.
Review/back up customized skills before replacing them. Restart Hermes if its
skill index does not yet show newly installed skills.

1. Hermes offers the optional signup link once; local forecasting needs no key.
2. After the human opts in, the agent loads `connect-ephemeris`.
3. Hermes's **native secure secret prompt** captures the key, outside the chat
   transcript. The signup link is also in the prompt's help text. Supported
   interactive Hermes clients use a masked prompt/secret overlay. Messaging
   clients and older clients without this feature must use the hidden terminal
   fallback; the agent must never ask for the key in chat.
4. Hermes stores `GNOMON_EPHEMERIS_API_TOKEN` in its active profile's secret store
   and passes the variable to terminal execution. The agent runs
   `gnomon connect ephemeris --from-env`, containing no secret in its arguments.
   Gnomon saves its private credential and discovers models without forecasting.
5. Reload MCP with Hermes `/reload-mcp` (or restart), then verify the new providers
   with `gnomon_capabilities`. The running MCP registry does not hot-reload.

The human signs up and enters the key; the agent handles the connection command.
No additional Gnomon credential tool is exposed to the model. This uses Hermes's
[secure setup-on-load contract](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/skills.md#secure-setup-on-load),
not a password field disguised as chat input. The backend must run on the same
host/user as Gnomon's MCP server; a remote sandbox's saved profile cannot
configure a separate local MCP process. An explicit provider TOML still wins.

`--from-env` reads only the fixed variable above, never a user-supplied variable
name, and never implicitly replaces an existing Gnomon connection. Replacement
requires `--replace`. Missing or invalid credentials fail without writing a
profile or making a network request. No automatic environment import happens
at startup. Declining/skipping setup leaves local providers available.

There are two credential copies: Hermes's profile secret and Gnomon's private
file. `--disconnect` removes only Gnomon's copy. Remove the Hermes secret through
its secret settings and revoke the key on Ephemeris for full removal. Neither
local store protects against arbitrary code running as the same OS user; the
flow keeps the key out of routine chat, tool arguments and tool results.

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
gnomon connect ephemeris --refresh-models # Update individual models; no forecast
gnomon connect ephemeris --status     # Local configuration status; no network
gnomon connect ephemeris --check      # Authenticated balance GET; no forecast
gnomon connect ephemeris --replace    # Hidden prompt to rotate the local key
gnomon connect ephemeris --disconnect # Remove local copy; restart MCP afterward
```

To revoke a key, use Ephemeris; removing a local copy does not revoke it.
Automation may use `--token-stdin` with a secure input source. Do not put a
literal token in a shell command, transcript, repository or MCP argument.

## CLI, Python and MCP

Connecting saves the key, then makes an authenticated `/models` GET. It caches
model names and capability flags, not arbitrary catalog metadata. New CLI
sessions and `GnomonSession.from_config()` load `ephemeris` (router),
`ephemeris/ensemble`, and healthy enabled individual models such as
`ephemeris/chronos2` from this saved catalog. Startup makes no catalog calls or
paid forecasts. Names come from the service, not a hardcoded model list.

Run `gnomon connect ephemeris --refresh-models` and restart MCP to update the
list. Availability is as of discovery, not a promise of current health; Gnomon
does not substitute another model when an explicit model fails. A failed
refresh preserves the old catalog. Failed discovery on a new/rotated key leaves
router and ensemble configured with a warning. Disconnect removes the saved
catalog as well as the key.
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
