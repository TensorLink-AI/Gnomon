# Shared Gnomon with optional Ditto memory

The `ditto` branch includes a working single-server MCP service in
[`services/hosted`](../../services/hosted). It shares forecasts, decisions,
observations, saved reviews and lessons across authenticated processes. Core
Gnomon remains dependency-free; installing hosting is optional.

The live acceptance probe has exercised Hermes's pinned MCP discovery, registry
and transport against this service and a dedicated Ditto workspace: agent A
records a forecast and decision, exits, a separate principal records actuals,
Gnomon scores them, exports a lesson, restarts, and agent B retrieves and checks
that lesson through a new Ditto connection. This is a deterministic integration
probe, not an LLM strategy-quality benchmark. See [validation](validation.md).

## Start a local service

From this branch's checkout (the unreleased core transaction API is required):

```bash
python -m venv .venv-hosted
.venv-hosted/bin/pip install . ./services/hosted
.venv-hosted/bin/gnomon-hosted --root ./shared-state init
.venv-hosted/bin/gnomon-hosted --root ./shared-state project-create --name research
```

Keep the returned project ID. Create separate principals with only the scopes
needed, substituting that ID below:

```bash
.venv-hosted/bin/gnomon-hosted --root ./shared-state token-create \
  --project PROJECT_ID --principal hermes-a \
  --permissions evidence.read,forecast.create,decision.create
.venv-hosted/bin/gnomon-hosted --root ./shared-state token-create \
  --project PROJECT_ID --principal outcome-ingestion --permissions actual.create
.venv-hosted/bin/gnomon-hosted --root ./shared-state token-create \
  --project PROJECT_ID --principal reviewer \
  --permissions evidence.read,decision.create,memory.export
.venv-hosted/bin/gnomon-hosted --root ./shared-state serve
```

Each command displays its new token once. Store it in a secret manager or private
file. The service stores only the SHA-256 verifier. Do not commit service data or
credentials. Default bind is loopback port 8765. Built-in forecasting needs no
external provider. Optional `project-create --providers /absolute/providers.toml`
loads operator-owned Gnomon providers (including configured Ephemeris adapters).
Remote callers cannot set provider configuration, read local paths, register
Python entrypoints, or set recording timestamps.

## Connect Hermes or another MCP client

The preferred URL is `http://127.0.0.1:8765/mcp` locally, or your TLS endpoint's
`/mcp` remotely. For clients requiring legacy SSE, use `/sse`; its `/messages/`
POST endpoint is authenticated and bound to the originating credential.

Hermes `config.yaml`:

```yaml
mcp_servers:
  gnomon:
    url: http://127.0.0.1:8765/mcp
    headers:
      Authorization: "Bearer ${GNOMON_SERVICE_TOKEN}"
    connect_timeout: 15
    tool_timeout: 90
```

Set `GNOMON_SERVICE_TOKEN` in the Hermes process environment. Each process can
use a different principal token for the same project. Reconnecting needs no
previous MCP session. There is no Hermes runtime dependency in the service.

The tools are `gnomon_forecast`, `gnomon_ledger`, `gnomon_memory` and
`gnomon_hosted`. The first three retain the core request shapes with the hosted
allowlist and explicit cutoffs. Mutations require a durable `idempotency_key`.
`gnomon_hosted` adds durable datasets, saved reviews, reference resolution,
request receipts, and explicit Ditto delivery/recall. See [tool examples](tools.md).

## Run computation beside Hermes

Use [local computation with shared evidence](local-computation.md) for local models,
local Gnomon/Ephemeris, or direct Ephemeris calls. The service accepts validated
forecast submissions, freezes evidence snapshots, and stores explicitly unverified
client analyses. The optional `gnomon-shared` CLI supports local forecasting and
scoring; the same operations are exposed through MCP.

## Optional Ditto connection

Create a dedicated Ditto workspace and claim its workspace-scoped API key. The
OAuth connection of an interactive chat is separate from the server credential.
Provide `DITTO_API_KEY` to the server process via your secret manager, systemd
`EnvironmentFile`, or the Compose `.env` file. It is never put in a ledger.

```bash
.venv-hosted/bin/gnomon-hosted --root ./shared-state ditto-configure \
  --project PROJECT_ID --token-env DITTO_API_KEY --graph YOUR_WORKSPACE_ALIAS
```

Every connection checks Ditto's reported default graph against that alias. A key
for another workspace is rejected. The connector discovers and validates the live
MCP tool schemas before use. Changing connection configuration creates a new
connection identity; old exports cannot silently move to the new destination.
Rotating a key for the same graph/environment name retains that identity.

Enqueue a lesson, then explicitly deliver it. No scheduler silently exports data.
A failed preflight remains pending; an ambiguous save becomes uncertain. Use
`export.reconcile` with the remote memory ID to fetch and compare its full content
before acknowledging it. The source/vendor ID is stable, but the service does not
assume a remote exactly-once guarantee. Ditto outages never roll back ledger data.
Shared local recall remains available through `gnomon_memory` without Ditto.

Recall is limited to acknowledged exports in the authorized project/connection.
It fetches full contents, compares their digest and references, enforces evidence
cutoffs, and recomputes a current review. Changed actuals are reported separately;
a saved lesson keeps its original evidence. Checked numbers do not validate the
narrative's causality or the real-world truth of submitted observations.

## Deployment and maintenance

See [operations](operations.md) for Compose/TLS, credentials, backup/restore,
retention, upgrade/rollback, and explicit workload limits.

- [Ownership and evidence contracts](contracts.md)
- [Initial persistence audit](persistence-audit.md)
- [Validation evidence](validation.md)
- [Durable reference schema](contracts/evidence-reference.schema.json)

Managed provisioning, unified Ditto login, additional database backends, automatic
policy changes and active-active deployment remain out of scope.
