# Encrypted off-server backups to R2

Keep the live SQLite ledgers on the server's persistent filesystem. R2 holds
immutable backup objects; it is not a shared filesystem or a live ledger backend.

The operator utility [r2_backup.py](../../services/hosted/deployment/r2_backup.py)
briefly stops the service, runs the existing consistent backup command, and restarts
it before encryption and upload. It encrypts the archive with **age**
using a public recipient, uploads under a unique timestamp/UUID key, downloads it
again and verifies the ciphertext SHA-256. A `last-success.json` receipt is written
only after that readback succeeds. Object keys are never intentionally reused.

The age private key belongs in an off-server secret store. Only its public recipient
is installed on the VM. R2 credentials should have Object Read & Write permissions
restricted to the dedicated backup bucket; public bucket access must be disabled.
The Cloudflare management connector does not necessarily have permission to create
those credentials. See [R2 authentication](https://developers.cloudflare.com/r2/api/tokens/).

## Installation

On a Linux Docker host, install `age`, Python and an isolated environment containing
`boto3`. Use Docker BuildKit/buildx to build the hosted image: the hosted Dockerfile
has its own ignore file, which legacy Docker builders do not honor. Pin the
resulting image by commit tag and record its immutable image ID.

Place the backup script at `/srv/gnomon/r2_backup.py`, and create a private
`/srv/gnomon/backup.env` (mode 0600) containing:

```dotenv
GNOMON_R2_ACCOUNT_ID=YOUR_ACCOUNT_ID
GNOMON_R2_BUCKET=YOUR_PRIVATE_BACKUP_BUCKET
GNOMON_R2_ACCESS_KEY_ID=YOUR_BUCKET_SCOPED_ACCESS_KEY_ID
GNOMON_R2_SECRET_ACCESS_KEY=YOUR_BUCKET_SCOPED_SECRET
GNOMON_BACKUP_RECIPIENT=age1YOUR_PUBLIC_RECIPIENT
```

Install the supplied [service](../../services/hosted/deployment/gnomon-backup.service)
and [timer](../../services/hosted/deployment/gnomon-backup.timer). Adjust container,
image, data and virtual-environment paths to your deployment. The example schedules
03:00 UTC daily with up to five minutes of jitter. It also starts the container
when the backup job exits, including on failures. Configure the main service to
start after VM reboot even if the VM stopped during a backup. The script's lock
prevents overlapping manual and scheduled runs.

Run and verify a manual backup before enabling the timer:

```bash
systemctl daemon-reload
systemctl start gnomon-backup.service
systemctl status gnomon-backup.service
cat /srv/gnomon/backup-work/last-success.json
systemctl enable --now gnomon-backup.timer
```

A failed upload leaves the previous last-success receipt unchanged and the service
running. Monitor failed systemd jobs, missing/old success receipts and disk usage.
This example does not configure external alert delivery or automatic retention
expiration. Before setting expiration, choose a recovery window and test restores.
Daily snapshots imply up to roughly a day of evidence loss, longer if jobs fail;
this is not a contractual recovery guarantee.

## Restore rehearsal

1. Download the object identified by a successful receipt and compare SHA-256.
2. On the recovery machine, decrypt with the separately stored age private key:

   ```bash
   age -d -i /private/backup.agekey -o backup.tar backup.tar.age
   ```

3. Extract the authenticated archive into a new private directory. It contains
   `evidence/manifest.json`, the databases and `deployment.json`.
4. Use the matching Gnomon build to restore into a **new** service directory:

   ```bash
   gnomon-hosted --root /srv/gnomon-restored restore --source ./evidence
   ```

5. Recover external configuration and credentials separately. Issue fresh agent
   credentials: restore revokes all backed-up credentials and marks pending exports
   uncertain. Start the isolated restored service and resolve known execution,
   snapshot and analysis references. Verify numerical results locally and check
   that old credentials are rejected.
6. Only after verification should an operator change the live endpoint. A rehearsal
   must not reconnect a second writable replica to the original data directory.

The backup includes evidence and a manifest, not external API credentials, provider
files or container images. Retain the matching source/build or image separately.
Restoration should be tested on another machine, not only against files on the VM
being backed up. Copying encrypted backups to R2 does not establish a working
restore until this procedure succeeds.
