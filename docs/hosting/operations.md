# Operating the single-server beta

Use one Linux server process on a local persistent filesystem. Each project has
one SQLite ledger; a control database maps project memberships and token verifiers.
Provider inference runs in a bounded subprocess outside ledger transactions.
Application receipts and core ledger effects commit in the same transaction.
A startup file lock rejects a second server using the same directory.

## Container and TLS

From the repository root:

```bash
docker compose -f services/hosted/deployment/compose.yaml build
docker compose -f services/hosted/deployment/compose.yaml run --rm gnomon init
docker compose -f services/hosted/deployment/compose.yaml run --rm gnomon project-create --name research
```

Issue tokens with `run --rm gnomon token-create ...` as in the quickstart, then
`docker compose -f services/hosted/deployment/compose.yaml up -d gnomon`.
The service runs as UID 10001, with a read-only root filesystem, dropped capabilities,
a writable named evidence volume, and a temporary `/tmp`. Do not run multiple
replicas against the volume. `/ready` and `/health` are unauthenticated liveness
checks; they do not certify provider or Ditto availability.

For remote access, point a DNS name at the host. Copy `.env.example` to a private
`services/hosted/deployment/.env`, set `GNOMON_HOSTNAME` to that name, and enable
the `tls` Compose profile. Supply Compose `--env-file` explicitly:

```bash
docker compose --env-file services/hosted/deployment/.env \
  -f services/hosted/deployment/compose.yaml --profile tls up -d
```

Caddy terminates TLS; the Gnomon port is published only on host loopback. Both
Host and browser Origin are validated. Allow an additional browser origin only
when needed, with `serve --allowed-origin https://your-client.example`. TLS
certificate issuance requires working DNS and inbound ports 80/443; local
container tests do not validate your DNS or public certificate setup.

Optional Ditto uses the dedicated workspace key in that private `.env`. Provider
configs and keys are operator-owned and need explicit mounts/environment variables.
Never bake credentials or ledgers into the image. Build from the matching branch:
this hosted package pins the core development version providing transaction support.

## Credentials and project retirement

To rotate a Gnomon token, issue a new token for the same principal and permission
set, update clients, then `token-revoke --token-id OLD_ID`. The old credential
fails on the next HTTP request and before a long-running call returns evidence.
Permission changes use a new principal; issuing a token cannot silently enlarge
existing tokens' membership. `project-disable --project ID` is a permanent access
tombstone in this beta: no re-enable command is exposed. Every subsequent request
and delivery fails authorization. In-flight external calls may already have run.

There is no automatic retention deletion or automatic export retry. Before removing
retired local project data, disable the project and reconcile in-flight exports.
Delete remote memories separately using their acknowledged memory IDs if required.
Keep the control tombstone; do not recreate its ID or replay an old outbox. Backup
restores revoke every token and mark unresolved exports uncertain so restoration
cannot silently repeat external writes. Restoring old backups of intentionally
retired projects requires reapplying their tombstones before issuing credentials.
This beta does not promise cross-system atomic deletion.

## Backup, restore, upgrade and rollback

Stop the service and all operator writers for a coordinated backup. The backup
command refuses to run while the server holds its lock:

```bash
gnomon-hosted --root /srv/gnomon backup --destination /backups/gnomon-2026-10-01
gnomon-hosted --root /srv/gnomon-restored restore --source /backups/gnomon-2026-10-01
```

The bundle contains consistent SQLite backups and a SHA-256 manifest with service
identity. Immutable dataset snapshots and review artifacts are inline in project
databases in this release. No external artifact store is needed. Provider TOML
paths are configuration references: separately back up/remap operator provider
files and recover their credentials securely. Remap a recovered config with
`provider-configure --project ID --providers /new/absolute/providers.toml`. Ditto keys are external secrets
and are never included. Store backups encrypted outside the host.

Restore into a new directory, recover provider configuration/secrets, issue new
tokens, then start the service and resolve known evidence references. Preserve
service/project/ledger IDs. The server URL may change; configure clients to use
the new address. Check all projects before resuming outcome ingestion or exports.

Before upgrade, stop writers, take and verify a backup, and restore it to a test
location with the new image. Schema version mismatches fail at startup; this beta
has one service schema and no implicit migrations. Roll back by restoring the
pre-upgrade bundle and the matching image, never by opening a newer database
with older code. Recovery time and data-loss tolerance depend on measured restore
time and backup frequency; no RPO/RTO guarantee is claimed.

## Failure states, observability and limits

- A principal's same idempotency key with identical arguments resolves its existing
  request. Different arguments conflict. Query a receipt after a lost response.
- Forecast provider failures/timeouts and unfinished operations found after a crash
  become `outcome_unknown`. They are not transparently re-executed. Inspect the
  provider before starting a distinct request. Synchronous timeout defaults to 60s,
  operator maximum 300s; hosted thread concurrency is bounded at eight.
- Ledger effects, immutable snapshots, request completion and the completion audit
  entry commit atomically. Saved reviews use one consistent evidence snapshot.
- Request bodies are limited to 1 MiB; forecast history to 10,000 and horizon to
  1,000 points; core batch limits also apply. Ditto operations have a 30s bound.
  Recall admits at most 100 eligible acknowledged exports per project/connection
  and returns at most five matches. Larger history requires explicit references
  or a future indexed recall design; it fails visibly rather than silently sampling.
- Operation receipts, export states and append-only audit rows are durable. Remote
  errors suppress provider details that could contain credentials. Uvicorn access
  logging is disabled. Monitor failed/unknown receipts, uncertain exports, disk
  usage, backup age, and provider/remote latency separately from HTTP readiness.

This is a bounded self-hosted beta, not active-active infrastructure or managed
customer provisioning. See measured probe results in [validation](validation.md).
