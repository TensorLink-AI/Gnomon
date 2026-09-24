---
name: connect-ephemeris
description: Connect optional Ephemeris models to Gnomon after the user opts in, using Hermes native secure credential entry.
required_environment_variables:
  - name: GNOMON_EPHEMERIS_API_TOKEN
    prompt: "Ephemeris API key (secure entry; blank skips setup)"
    help: "Sign up and create an API key at https://ephemeris.cascade.industries. Paste it only in this secure prompt, never chat."
    required_for: Optional Ephemeris connection, not local forecasting
---

# Connect Ephemeris in Hermes

Load this skill only after the user accepts optional Ephemeris signup. Offer it
once; respect a decline. Show https://ephemeris.cascade.industries first so the
human can sign up and obtain a key. Never ask for a key in a chat message.

Hermes handles this skill's declared credential with its native secure prompt.
On supported interactive surfaces this is a masked prompt or secret overlay,
outside the conversation. Hermes persists it in the active profile's secret
store and passes it to terminal execution. The skill/tool response contains
setup status, not the key. Do not read, echo, log, interpolate or inspect the key,
the environment, or Hermes `.env`. Do not pass it as an MCP argument.

If setup is skipped, missing, or the host lacks secure entry, stop setup and
continue local forecasting. Offer the existing hidden terminal command
`gnomon connect ephemeris` as an alternative. Never fall back to chat entry.

After successful native capture, execute exactly:

```sh
gnomon connect ephemeris --from-env
```

This command reads only GNOMON_EPHEMERIS_API_TOKEN internally. It saves a private
Gnomon credential and discovers models with an authenticated GET, without
forecasting. There is no token in the command or returned JSON. An existing
Gnomon connection is preserved; use `--replace` only when the user explicitly
requests replacement. A discovery failure is not proof of valid credentials;
follow the returned guidance without printing secrets.

Run this in the same host/user environment as the Gnomon MCP server. For a remote
terminal/sandbox, stop if it differs: saving there will not configure a local
MCP server. Do not copy secrets between hosts through tool results.

Then reload Gnomon's MCP server through the host's supported reload mechanism
(Hermes `/reload-mcp`, or restart Hermes if unavailable). The existing MCP
process does not see newly registered providers until reloaded. Call
gnomon_capabilities again and verify configured_in_session and provider names.
Only then report connection complete. If it uses an explicit providers TOML,
that overrides the saved profile: report this and do not rewrite operator config.

The available names include `ephemeris` (router), `ephemeris/ensemble`, and
discovered `ephemeris/<model>` entries. Connecting does not authorize paid
forecasts. Ask for task authorization before billable execution.

Disconnecting Gnomon removes its copy, not Hermes's saved secret. To fully remove
access, remove GNOMON_EPHEMERIS_API_TOKEN through Hermes secret settings and
revoke the key on Ephemeris; never read the secret back into the conversation.
