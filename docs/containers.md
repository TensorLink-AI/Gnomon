# Containers

## Build locally

```bash
docker build -t aion .
docker run --rm aion capabilities
```

The image runs as an unprivileged `aion` user and uses the CLI as its
entrypoint.

## Forecast mounted data

```bash
mkdir -p aion-output
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/examples:/input:ro" \
  -v "$PWD/aion-output:/output" \
  aion forecast /input/daily_requests.csv \
    --time timestamp \
    --target requests \
    --horizon 3 \
    --frequency D \
    --output /output
```

Mount inputs read-only and use a separate writable output mount. Mapping the
container process to the host user makes the resulting artifacts writable by
that user. Without `--user`, the image defaults to its unprivileged internal
`aion` account.

## GitHub Container Registry

The container workflow builds pull requests without publishing. Pushes to
`main` publish branch and `latest` tags; version tags such as `v0.1.0` publish
semantic-version tags to:

```text
ghcr.io/tensorlink-ai/aion
```

After the first workflow succeeds:

```bash
docker pull ghcr.io/tensorlink-ai/aion:latest
docker run --rm ghcr.io/tensorlink-ai/aion:latest capabilities
```

Package visibility is controlled in the GitHub organization/package settings.
