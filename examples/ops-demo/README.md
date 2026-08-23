# Operations loop demo

This reference deployment is intentionally small: a metric exporter, a local
Prometheus, and one cron-shaped Gnomon monitor invocation. It demonstrates the
complete boundary—range query in, immutable report, durable event, webhook,
and a Prometheus-compatible static threshold rule out.

```bash
docker compose -f examples/ops-demo/compose.yaml up --build \
  --abort-on-container-exit --exit-code-from monitor
docker compose -f examples/ops-demo/compose.yaml logs monitor
```

The run waits about 45 seconds to accumulate a one-second history. Named
volumes retain `/output`, idempotency state, and received webhook events.
Replaying the same monitor artifact records the same content-addressed event
without sending a duplicate; a later forecast can correctly produce a new
event. The demo receiver and one-second scrape interval are not a production
deployment; production callers should run `gnomon monitor run` from their
scheduler and retain the state file on durable storage.
