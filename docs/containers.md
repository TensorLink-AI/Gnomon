# Containers

```bash
docker build -t gnomon .
docker run --rm gnomon capabilities
docker run --rm -v "$PWD/examples:/input:ro" gnomon infer \
  --input /input/daily_requests.csv --target-column requests \
  --provider last_value --horizon 3
```

The image runs an unprivileged user and uses the CLI as its entrypoint.
Mount inputs read-only. A persistent ledger needs a separate writable mount and
explicit provider configuration with its ledger path. Match host/container file
ownership when supplying that mount.

Pull-request CI builds without publishing. Main/version-tag publication is a
separate release action; a successful build does not mean an image was published.
Use an approved immutable digest for deployment, not an assumed current latest tag.

The old Prometheus-to-webhook monitoring demo is retired. Gnomon records evidence;
external scheduling, model serving and action execution belong to the host.
