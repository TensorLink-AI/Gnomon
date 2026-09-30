# Gnomon evidence recall for Hermes

This optional plugin adds project-scoped ledger evidence before matching Hermes
turns. It works whether the agent forecasts through the direct CLI or Gnomon's
MCP server: the hook reads the same ledger with `gnomon memory`.
It uses Hermes's [documented pre_llm_call plugin hook](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins/).

## Quickstart

You need Gnomon installed, Hermes with `pre_llm_call` plugin support, and an
existing Gnomon ledger containing reviewed lessons. An empty ledger returns no
memories. To learn how to create decisions and lessons, use the
[ledger guide](https://github.com/TensorLink-AI/Gnomon/blob/main/docs/hermes-ledger.md).

1. **Install the plugin once.** From your Gnomon repository directory, run:

   ```bash
   mkdir -p "${HERMES_HOME:-$HOME/.hermes}/plugins"
   cp -Rn integrations/hermes/gnomon-memory "${HERMES_HOME:-$HOME/.hermes}/plugins/"
   hermes plugins enable gnomon-memory
   ```

   For a packaged installation, the plugin is under
   `share/gnomon/integrations/hermes/gnomon-memory` in your installation prefix.
   Copy that directory instead. The command above preserves an existing copy;
   it is a first-install command, not an upgrade command.

2. **Create the two configuration files below** in your project. Replace the
   example project path, Gnomon executable, series, unit, horizon and cutoffs
   with your own. `command -v gnomon` shows the executable in your active
   environment. Use the ledger already used by your CLI or MCP configuration.

3. **Check that Gnomon can read your ledger.** From the project directory:

   ```bash
   gnomon memory --providers-config providers.toml --arguments '{"operation":"recall","series_id":"sales","unit":"widgets","horizon":2,"source_as_of":"2026-01-23T00:00:00Z","recorded_as_of":"2026-01-23T00:00:00Z"}'
   ```

   Use the same scope and cutoffs as your configuration. Expect `status: ok`;
   `returned` tells you how many lessons matched. Zero is a successful read
   with no matching lessons, so nothing will be injected into Hermes.

4. **Launch a fresh Hermes session with the configuration enabled:**

   ```bash
   cd /work/sales
   export GNOMON_MEMORY_CONFIG="$PWD/gnomon-memory.json"
   hermes
   ```

   Set this variable in the launch environment again for future sessions.

5. **Ask a matching question**, such as “Review the ledger evidence for our
   sales forecast.” The word `sales` triggers the example configuration.
   Relevant saved evidence is added before the turn automatically. You can
   continue using either CLI or MCP tools for forecasting.

The optional compatibility job tests real Hermes plugin discovery, hook dispatch,
context composition and schema sanitization at the commit in `hermes-revision.txt`.
Support is currently project-local CLI sessions; gateway and multi-project
sessions are excluded until session-to-project scoping is implemented and tested. For a new installation, check the
direct read in step 3 before diagnosing the hook.

## Project configuration

Create an operator-owned JSON file, e.g. `/work/sales/gnomon-memory.json`:

```json
{
  "project_root": "/work/sales",
  "providers_config": "providers.toml",
  "command": ["/absolute/path/to/venv/bin/gnomon"],
  "match_terms": ["sales", "retail forecast"],
  "cutoff_mode": "fixed",
  "query": {
    "operation": "recall",
    "series_id": "sales",
    "unit": "widgets",
    "horizon": 2,
    "source_as_of": "2026-01-23T00:00:00Z",
    "recorded_as_of": "2026-01-23T00:00:00Z",
    "limit": 2,
    "max_context_chars": 6000
  }
}
```

Set `GNOMON_MEMORY_CONFIG=/work/sales/gnomon-memory.json` in the environment
launching Hermes. Launch from that project directory. The hook only runs within
`project_root` and when the user message contains a configured match term
(case-insensitive literal match). It does not infer instruments or share recall
across projects. Use distinct configurations for distinct tasks.

`providers.toml` needs an existing ledger:

```toml
ledger_path = "evidence.db"
[memory]
ledger_ref = "sales-project"
```

Choose actual evaluation cutoffs for historical work. For explicitly prospective
live use, set `cutoff_mode` to `live`; both cutoffs then become the current UTC
instant at the start of the hook. Never use live mode for a historical evaluation.
Configuration is trusted operator input; the hook never takes a command or path
from the user's message. `providers_config` is relative to the JSON file.

The hook recalls up to three saved lessons for the exact series/unit/horizon,
rechecks them against visible evidence, and injects at most 6000 context
characters plus a short label. Oversized records are omitted with truncation
reported by the CLI. Empty recall injects nothing. Failure logs a warning and
allows the turn to proceed without recalled evidence. The subprocess has a
10-second timeout. No forecast providers are loaded during the read.

This is ephemeral context, not a native Hermes memory write. Record decisions,
actuals and reviewed lessons explicitly through `gnomon_ledger` / `gnomon ledger`.
The existing Python `HermesMemoryAdapter` can propose explicit native-memory
updates separately. This plugin neither delivers those updates nor fabricates
lessons. Use the ledger review operations to inspect exact evidence.

Disable with `hermes plugins disable gnomon-memory` or unset
`GNOMON_MEMORY_CONFIG`. No global memory files are modified.

## If nothing appears

- **The direct read returns zero:** check that saved lessons exist and that
  series, unit, horizon and evidence cutoffs match them.
- **The direct read works but Hermes adds nothing:** check the plugin is enabled,
  restart Hermes with the environment variable set, launch inside `project_root`,
  and include a configured match term in your message.
- **“Gnomon recall unavailable” appears in the log:** check the absolute executable
  path and configuration paths. Run step 3 using the configured executable.
- **You expected lessons to be created automatically:** this plugin recalls
  existing lessons. Recording decisions, outcomes and reviewed lessons remains
  an explicit ledger operation.
