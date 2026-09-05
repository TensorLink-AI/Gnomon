# Ordinary-agent software environment

This optional benchmark backend gives agents ordinary Python and real forecasting
software. It is not a Gnomon provider adapter, core dependency or proof of agent
improvement. Agents use installed APIs directly, inspect library documentation,
calculate with the standard library and save/load files between calls. No Gnomon
package is installed in this image.

The hash lock targets Linux x86-64 / Python3.12: StatsForecast2.1.1, pandas2.3.3,
NumPy2.5.2, SciPy1.18.1 and exact transitive dependencies. Installation accepts only
hashed wheels, with no source builds. These are exercised versions, not a claim
that they are best for every task. Other host Python versions can run the driver;
container Python stays pinned. Other architectures need independent verification.

## Build and use

Explicitly build with a local Docker daemon:

```bash
docker build --pull=false -t gnomon-bench-software:local benchmarks/workflow/software
docker image inspect gnomon-bench-software:local --format '{{.Id}}'
```

Dockerfile pins the base by repository digest. Its build context includes only the
Dockerfile and lock, never repository code/cases/credentials. Package downloads
happen during this explicit build, not agent calls. Installation verifies hashes
and runs `pip check`. If normal Docker client configuration is read-only, use
`docker --config /path/to/new/private-directory`; no registry login is needed for
these public inputs. Build metadata can change the resulting manifest ID; record
the actual installed ID, not a tag or an assumed reproducible image checksum.

In the [shared driver configuration](../MATCHED.md), use this ordinary entry:

```json
{
  "factory": "benchmarks.workflow.software_backend:make",
  "options": {
    "image": "sha256:REPLACE_WITH_THE_ACTUAL_64_HEX_IMAGE_ID",
    "docker_host": "unix:///var/run/docker.sock"
  }
}
```

The placeholder is intentionally invalid until replaced. Only already-installed
SHA256 image IDs and explicit local Unix sockets are accepted. The backend never
pulls images, loads remote Docker contexts, mounts the checkout or forwards the
host environment. Include Dockerfile, requirements.in and requirements.txt in
`driver_files`; backend Python is already covered by the checkout pin. Result
provenance records image ID, Python/distribution versions, public-input hash and
limits. Installed metadata is supplied provenance, not a supply-chain audit.

`python` is the sole tool, with argument `{ "code": "ordinary Python source" }`.
Public data is `/tmp/case.json`. Print results and save work under `/tmp`. Each call
has fresh globals; files persist between calls. Python exceptions return stderr
and a nonzero return code so the agent can repair them. No code/answer rewrite,
forced model choice or automatic retry occurs. Future observations are not
preloaded; the driver still accepts only public single-stage tasks.

Optional public `available_at_cutoff.files` maps at most32 simple filenames to
UTF-8 text, materialized under `/tmp/data`. No host reads, absolute paths or parent
traversal are accepted; the whole public input stays within the existing byte cap.

## Resource and security boundaries

- Agent code runs as UID/GID65534, with no capabilities or privilege escalation,
  no network or host mounts, a read-only root and128MiB `/tmp` tmpfs.
- Limits are1GiB memory (swap capped at the same total),128 processes and two CPUs.
  Responses are bounded during capture to512KiB stdout/64KiB stderr; input code
  is at most64KiB. Keep these fixed limits equal for ordinary access in every arm.
- A minimal trusted UID0 PID1 timer with dropped capabilities exits after the
  lifetime. Non-root agent code cannot signal it. Namespace termination ends
  descendants even if the host driver disappears. Startup counts toward lifetime;
  OS scheduling still adds overhead. A crash between create/start can leave a
  stopped container; this is not a hard real-time or zero-residue guarantee.
- Timeout/output overflow removes the owned container, not merely the host Docker
  client. Close verifies a random ownership label and exact ID before removal.
  It never prunes or deletes by wildcard. An unavailable daemon is not reported
  as successful cleanup. Retain exact ownership identity for manual recovery.
- Docker/image/daemon/kernel code is trusted. This is not VM isolation or a
  certified hostile-code sandbox; use a dedicated runner for untrusted workloads.
  Image authors must not bake in credentials. No host credentials are forwarded.
- Saved files are ephemeral: cleanup/timeout destroys them. This is not the
  production temporal ledger or durable agent checkpoint. Local tool service
  charges are0; infrastructure/energy is excluded, not described as free. LLM
  billing and remote request cancellation remain separate.

## Verification and remaining work

```bash
GNOMON_TEST_SOFTWARE_IMAGE=$(docker image inspect gnomon-bench-software:local --format '{{.Id}}')
export GNOMON_TEST_SOFTWARE_IMAGE
python -m pytest -q benchmarks/tests/test_software_backend.py
```

Without that explicit variable, container tests skip rather than discover/build
an image. A dedicated CI job explicitly builds and runs them. Checks execute real
SeasonalNaive/AutoARIMA/AutoETS forecasts, file/error recovery, permission/network/
credential boundaries, resource settings, timeout/output cleanup, ownership refusal
and independent timer expiry. A scripted model also uses the backend through the
shared loop. These are software/lifecycle checks, not LLM-quality measurements.

The [lean/full backends](../service/README.md) now retain this software in one
container and the actual Gnomon service in another. Extra compute and different
forecast contracts are explicit. Appropriate tasks, staged agent-owned forecast/
ledger journeys and total paid-run spending controls remain required before the
comparison earns acceptance credit.
