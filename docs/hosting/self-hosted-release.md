# Self-hosted Gnomon MCP beta

This beta provides shared forecasts, decisions, actuals, reviews and lessons over
an authenticated MCP endpoint. Local Gnomon or Ephemeris can do the computation;
the server can also run built-in forecasts. Ditto is optional. The initial image
supports Linux amd64 and one server instance per persistent evidence volume.

## Install from a release

Install Docker Engine and Docker Compose 2.24 or newer. Download `compose.yaml`,
`Caddyfile`, `env.example`, and `image-digest.txt` from the same `hosted-v*` GitHub
prerelease into an empty directory. No Python installation is needed.

```bash
cp env.example .env
chmod 600 .env
```

For reproducible deployment, set `GNOMON_IMAGE` in `.env` to the exact
`ghcr.io/tensorlink-ai/gnomon-hosted@sha256:...` reference from `image-digest.txt`.

```bash
docker compose pull gnomon
docker compose run --rm gnomon init
docker compose run --rm gnomon project-create --name research
```

Copy the returned project ID:

```bash
PROJECT_ID='project-REPLACE_WITH_RETURNED_ID'
docker compose run --rm gnomon token-create --project "$PROJECT_ID" \
  --principal agent-a --permissions evidence.read,forecast.create,decision.create,memory.export
docker compose run --rm gnomon token-create --project "$PROJECT_ID" \
  --principal agent-b --permissions evidence.read,forecast.create,decision.create,memory.export
docker compose run --rm gnomon token-create --project "$PROJECT_ID" \
  --principal outcomes --permissions evidence.read,actual.create,decision.create
docker compose up -d gnomon
curl --fail http://127.0.0.1:8765/ready
```

Save each returned token privately, and its token ID for revocation. Set each
agent's `GNOMON_SERVICE_TOKEN` environment variable to its own token. The outcomes
process has a separate credential for appending observations and evaluating them.

For Hermes, merge this into its configuration:

```yaml
mcp_servers:
  gnomon:
    url: http://127.0.0.1:8765/mcp
    headers:
      Authorization: "Bearer ${GNOMON_SERVICE_TOKEN}"
    connect_timeout: 15
    tool_timeout: 90
```

Other clients use the same MCP URL and bearer header. The loopback URL assumes
that the client runs on the Docker host. First call `gnomon_hosted` with
`{"action":"info"}` to verify the authenticated project and permissions.

## Image archive alternative

Each prerelease also includes `hosted-image.tar.gz` and its checksum. This works
without a registry login and can be transferred to a disconnected host. Download
both from the same release, verify and load them:

```bash
sha256sum -c hosted-image.tar.gz.sha256
docker load -i hosted-image.tar.gz
```

Set `GNOMON_IMAGE` in `.env` to the versioned image name printed by `docker load`
(rather than the registry digest), skip `docker compose pull`, and follow the same
initialization commands. Optional Ditto and public TLS still need network access.

## Remote agents and HTTPS

Point your domain's DNS at the server and allow inbound TCP ports 80 and 443.
Set `GNOMON_HOSTNAME` in `.env` to that domain, then:

```bash
docker compose --profile tls up -d
```

Caddy obtains TLS certificates. Agents on other machines connect to
`https://YOUR_DOMAIN/mcp` with their own bearer tokens. Certificate issuance
requires working DNS and network access. Gnomon's application port stays bound
to host loopback; clients in separate containers also need a reachable hostname.

## Optional Ditto

Shared evidence works without Ditto. To enable lesson export and recall, obtain
a dedicated Ditto workspace key and put `DITTO_API_KEY` in the private `.env`.
Then configure the project and recreate the application to load that environment:

```bash
docker compose run --rm gnomon ditto-configure --project "$PROJECT_ID" \
  --token-env DITTO_API_KEY --graph YOUR_WORKSPACE_ALIAS
docker compose up -d --force-recreate gnomon
```

This does not automatically export every forecast. Record a decision, ingest
actuals, create a snapshot and submit an analysis with `export_to_ditto: true`,
then call `export.deliver`. Keep retrospective backtests labelled as such.

## Persistence, backups and upgrades

The `evidence` named volume survives container replacement and `docker compose down`.
Do not use `down -v` unless you intend to erase it. Keep the same Compose project
name/directory across upgrades so you reuse the existing volume. Run one server
instance against that volume.

Before upgrading, stop writers and take a backup into a fresh named volume:

```bash
docker compose stop gnomon
docker volume create gnomon-backup-before-upgrade
docker compose run --rm --user 0 \
  -v gnomon-backup-before-upgrade:/backup gnomon backup --destination /backup/evidence
docker compose start gnomon
```

Copy backups off the machine and store them encrypted. The backup volume alone
is not disaster recovery. Provider/Ditto secrets need separate secure recovery.
Restore into a NEW evidence volume and issue fresh credentials; restores revoke
old tokens. For a recovery rehearsal, choose a fresh Compose project name:

```bash
docker compose -p gnomon-restore run --rm --user 0 --entrypoint sh \
  -v gnomon-backup-before-upgrade:/backup:ro gnomon -c \
  'gnomon-hosted --root /data restore --source /backup/evidence && chown -R 10001:10001 /data'
```

After verifying that restore succeeds, resolve known records using fresh tokens
in an isolated instance before cutting over. Avoid running the restored instance
on the live instance's ports. Preserve the old image digest and backup for rollback.
This beta has no automatic schema migrations or active-active failover.

R2 backup scheduling is an optional operator setup, not enabled by this image.
The repository's `docs/hosting/operations.md`, `docs/hosting/tools.md`, and
`docs/hosting/r2-backups.md` cover detailed operation and evidence contracts.
