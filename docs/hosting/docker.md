# Self-host Gnomon MCP with Docker

This beta runs an authenticated remote MCP server with durable project evidence.
Hermes and other MCP clients connect over HTTP locally or HTTPS remotely. Models
can run on the server or beside the client. Ditto and R2 are optional integrations;
shared forecasts, decisions, actuals, reviews and lessons work without them.

Use a checkout of the `ditto` branch, Docker Engine with BuildKit/buildx, and Docker
Compose 2.24 or newer. Build from the repository root. There is no published image
for this beta; these commands build the image from your checkout.

## Build and initialize

```bash
docker compose -f services/hosted/deployment/compose.yaml build
docker compose -f services/hosted/deployment/compose.yaml run --rm gnomon init
docker compose -f services/hosted/deployment/compose.yaml run --rm gnomon \
  project-create --name research
```

Copy the returned project ID into this variable, then create a credential for each
agent. Each token command prints a JSON object containing `token` and `token_id`.
Keep the token private; keep its ID for later revocation.

```bash
PROJECT_ID='project-REPLACE_WITH_RETURNED_ID'
docker compose -f services/hosted/deployment/compose.yaml run --rm gnomon \
  token-create --project "$PROJECT_ID" --principal hermes-a \
  --permissions evidence.read,forecast.create,decision.create,memory.export
docker compose -f services/hosted/deployment/compose.yaml run --rm gnomon \
  token-create --project "$PROJECT_ID" --principal hermes-b \
  --permissions evidence.read,forecast.create,decision.create,memory.export
docker compose -f services/hosted/deployment/compose.yaml run --rm gnomon \
  token-create --project "$PROJECT_ID" --principal outcome-ingestion \
  --permissions actual.create
docker compose -f services/hosted/deployment/compose.yaml up -d gnomon
curl --fail http://127.0.0.1:8765/ready
```

Agents share a project but have separately revocable credentials. The outcome
process uses its own token to record actuals. Agent permissions do not authorize
outcome ingestion. `memory.export` permits export when an operator configures Ditto;
it does not set up a Ditto connection automatically.

## Connect a client

Set `GNOMON_SERVICE_TOKEN` in each agent process to its own token. For Hermes,
merge this entry into `mcp_servers` in its configuration:

```yaml
mcp_servers:
  gnomon:
    url: http://127.0.0.1:8765/mcp
    headers:
      Authorization: "Bearer ${GNOMON_SERVICE_TOKEN}"
    connect_timeout: 15
    tool_timeout: 90
```

The loopback URL works for a client on the Docker host. For another machine, use
the HTTPS setup below. A client in another container needs a reachable service
hostname and the corresponding server `--allowed-host` setting.

First call `gnomon_hosted` with `{"action":"info"}` to check project identity and
permissions. Use `gnomon_forecast` to execute a configured server model. Use
`gnomon_hosted` with `action: "forecast.submit"` to record a forecast already
computed locally or through Ephemeris. Both require a named series, history and
target timestamps, and an idempotency key. See [tool examples](tools.md) and
[local computation](local-computation.md). Ephemeris is not configured by default.

## Expose HTTPS

Point a domain at the Docker host and allow inbound ports 80 and 443. Copy
`services/hosted/deployment/.env.example` to `services/hosted/deployment/.env`,
restrict the file to its owner, and set `GNOMON_HOSTNAME` to your domain:

```bash
cp services/hosted/deployment/.env.example services/hosted/deployment/.env
chmod 600 services/hosted/deployment/.env
# Edit GNOMON_HOSTNAME before starting the TLS profile.
docker compose --env-file services/hosted/deployment/.env \
  -f services/hosted/deployment/compose.yaml --profile tls up -d
```

Caddy obtains a certificate and proxies `https://YOUR_DOMAIN/mcp`. Use that URL
in each client; bearer authentication is still required. The application port
remains bound to host loopback. Certificate issuance depends on your DNS and
network configuration.

## Storage and maintenance

The Compose `evidence` volume survives container recreation and `compose down`.
`compose down -v` deletes that volume. Run one server instance per evidence volume.
The application runs as UID 10001 with a read-only root filesystem.

Back up the evidence before upgrading, and use a matching image for recovery.
See [operations](operations.md) for backup/restore, credential rotation and
operator-owned provider configuration, and [R2 backups](r2-backups.md) for optional
encrypted off-server backups. The image does not start an R2 backup schedule.

The [Ditto setup](README.md#optional-ditto-connection) is optional. Self-hosting
requires neither the Cascade staging service nor its credentials.
