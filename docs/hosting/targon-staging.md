# Targon staging deployment

Recorded 2026-10-01. The dedicated `gnomon-shared-staging` CPU VM runs commit
`68c0f2c9912403f27c43be570b71a19ed59323a2`, tagged `gnomon-hosted:68c0f2c9`.
The [sanitized deployment report](evidence/targon-staging.json) includes immutable
image identity and acceptance evidence. Observed Targon rental cost is $0.12/hour.
No production release has been published.

## Verified

- Persistent data at `/srv/gnomon/data`, unprivileged read-only application container,
  Gnomon token authentication, separate agent principals and project isolation.
- Cloudflare Tunnel connected from the VM; application port bound to loopback.
- Separate pinned Hermes installations on the client machine and VM exchanged
  shared evidence using actual MCP discovery/dispatch over public HTTPS. The
  original SSH test also passed. Unauthenticated public MCP requests return 401
  and another project cannot resolve the shared execution.
- Local forecast → shared execution/decision → authorized actual → immutable
  snapshot → local analysis → actual Ditto export → fresh remote Hermes recall.
  A synthetic forecast of 12 and actual of 13 produced locally recomputed MAE 1.
- Full VM reboot changed the boot ID; both containers restarted automatically,
  the ledger survived, and the remote Hermes installation recalled its lesson.
- An age-encrypted backup was transferred by SSH to a different machine, decrypted
  using an off-VM key and restored into an isolated new service directory. Original
  execution, snapshot and analysis references resolved; old tokens were rejected.

## Public access and R2 validation

Retested 2026-10-01: `https://gnomon-staging.cascade.industries/mcp` works with
both independent pinned Hermes clients. Readiness returns 200; unauthenticated MCP
returns 401. The earlier Cloudflare 1010 blocker no longer occurs. This retry made
no Cloudflare security configuration changes and does not establish which external
configuration change resolved it.

The encrypted backup uploaded to private `gnomon-staging-backups` and passed a
SHA-256 check after download. A separate machine downloaded the R2 object,
decrypted it using the off-VM key, restored the evidence into an isolated service,
resolved the execution/snapshot/analysis, recomputed MAE 1 and rejected old tokens.
The daily timer is now enabled for 03:00 UTC with up to five minutes of jitter.
The earlier failed upload recorded no successful backup; the successful receipt is
included in the deployment report.

## Operator locations

On the VM: `/srv/gnomon/deployment.json`, `/srv/gnomon/clients.json`,
`/srv/gnomon/server.env`, `/srv/gnomon/r2_backup.py`; units are
`gnomon-staging.service`, `gnomon-backup.service` and `gnomon-backup.timer`.
The backup timer is enabled after a successful upload/readback and separate-machine
restore test. The service remains healthy after the backup.

Off the VM, private recovery material and separate Hermes environment files are
stored in `/root/Gnomon/results/gnomon-staging-private/` (git-ignored, mode 0700).
The age private key is `backup.agekey`; keep an additional copy in your own secure
secret store. Credential values are excluded from repository documentation.
The private directory also contains an SSH profile for operator access. External provider and Ditto keys require separate
recovery; they are not in evidence backups.

For backups and recovery procedures, see [encrypted R2 backups](r2-backups.md).
Before calling this release-ready, run sustained use checks. This proves a synthetic evidence handoff, not forecast quality or
fully autonomous Hermes learning.
