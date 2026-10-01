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
  shared evidence using actual MCP discovery/dispatch. The current network path
  was encrypted SSH plus VM loopback, not the public HTTPS endpoint.
- Local forecast → shared execution/decision → authorized actual → immutable
  snapshot → local analysis → actual Ditto export → fresh remote Hermes recall.
  A synthetic forecast of 12 and actual of 13 produced locally recomputed MAE 1.
- Full VM reboot changed the boot ID; both containers restarted automatically,
  the ledger survived, and the remote Hermes installation recalled its lesson.
- An age-encrypted backup was transferred by SSH to a different machine, decrypted
  using an off-VM key and restored into an isolated new service directory. Original
  execution, snapshot and analysis references resolved; old tokens were rejected.

## Pending before public staging sign-off

1. `https://gnomon-staging.cascade.industries/mcp` currently receives Cloudflare
   Browser Integrity Check error 1010 before reaching Gnomon. A configuration rule
   limited to that hostname was rejected by automatic approval review. Explicit
   user approval is pending. No zone-wide security settings were changed.
2. The private `gnomon-staging-backups` R2 bucket exists with public access disabled.
   Scoped Object Read & Write credentials are still required. The management
   connector cannot create API tokens. No R2 upload has been claimed or scheduled.
   After credentials are installed, run upload/readback and restore from the R2
   object, then enable the daily timer. The SSH restore is not an R2 restore test.

## Operator locations

On the VM: `/srv/gnomon/deployment.json`, `/srv/gnomon/clients.json`,
`/srv/gnomon/server.env`, `/srv/gnomon/r2_backup.py`; units are
`gnomon-staging.service`, `gnomon-backup.service` and `gnomon-backup.timer`.
The backup timer is installed but disabled pending credentials and a successful run.

Off the VM, private recovery material and separate Hermes environment files are
stored in `/root/Gnomon/results/gnomon-staging-private/` (git-ignored, mode 0700).
The age private key is `backup.agekey`; keep an additional copy in your own secure
secret store. Credential values are excluded from repository documentation.
The private directory also contains an SSH profile for accessing staging while the
public hostname is blocked. External provider and Ditto keys require separate
recovery; they are not in evidence backups.

For backups and recovery procedures, see [encrypted R2 backups](r2-backups.md).
Before calling this release-ready, complete the two pending items and run sustained
use checks. This proves a synthetic evidence handoff, not forecast quality or
fully autonomous Hermes learning.
