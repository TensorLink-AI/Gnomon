# Containers

## Build locally

```bash
docker build -t gnomon .
docker run --rm gnomon capabilities
```

The image runs as an unprivileged `gnomon` user and uses the CLI as its
entrypoint.

## Forecast mounted data

```bash
mkdir -p gnomon-output
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/examples:/input:ro" \
  -v "$PWD/gnomon-output:/output" \
  gnomon forecast /input/daily_requests.csv \
    --time timestamp \
    --target requests \
    --horizon 3 \
    --frequency D \
    --output /output
```

Mount inputs read-only and use a separate writable output mount. Mapping the
container process to the host user makes the resulting artifacts writable by
that user. Without `--user`, the image defaults to its unprivileged internal
`gnomon` account.

## GitHub Container Registry

The container workflow builds pull requests without publishing. Pushes to
`main` publish branch and `latest` tags; version tags such as `v0.1.0` publish
semantic-version tags to:

```text
ghcr.io/tensorlink-ai/gnomon
```

After the first workflow succeeds:

```bash
docker pull ghcr.io/tensorlink-ai/gnomon:latest
docker run --rm ghcr.io/tensorlink-ai/gnomon:latest capabilities
```

Package visibility is controlled in the GitHub organization/package settings.

## Prometheus-to-webhook reference deployment

The bounded operations demo exercises a real Prometheus range query, an
integrity-sealed monitor artifact, durable idempotency state, webhook delivery, and
Prometheus rule export:

```bash
docker compose -f examples/ops-demo/compose.yaml up --build --abort-on-container-exit
```

See [the demo runbook](../examples/ops-demo/README.md). The generic integration
escape hatch is CSV over stdin (`export-command | gnomon forecast - ...`);
Gnomon deliberately does not execute an agent-supplied shell command.
