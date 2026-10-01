# Hosted service validation

Recorded 2026-10-01 on the `ditto` branch, Python 3.13.11 and MCP SDK 1.30.0.
These include synthetic integration tests and a prospective live measurement.
They are not evidence of better forecasts or autonomous trading performance. Historical [milestone A storage results](milestone-a-validation.md)
remain available separately.

## Live Hermes → Gnomon → Ditto → Hermes

The opt-in [live probe](../../services/hosted/scripts/live_smoke.py) completed with
actual Ditto save/search/fetch calls using a key for a dedicated test workspace.
It used the pinned Hermes checkout's discovery, schema sanitation, registry dispatch
and MCP transport in fresh OS processes. There were no LLM calls.

1. Hermes A recorded a forecast of 12 widgets and a decision before the target time.
2. That client exited and the Gnomon server stopped.
3. A separately authorized outcomes client supplied a synthetic actual of 13.
4. Gnomon saved a review (MAE 1) and an explicitly labelled synthetic lesson.
5. A durable export was acknowledged by the real Ditto endpoint.
6. Gnomon stopped and restarted again.
7. A fresh Hermes B process recalled through a new Ditto session. The connector
   fetched full memory contents and checked the payload digest, project reference,
   cutoff eligibility and core ledger review. It returned MAE 1 and
   `narrative_verified=false`.

The sanitized [live report](evidence/live-hermes-ditto.json) records the resource
IDs; IDs confer no access. The test's private service directory and credentials
are not committed. Reproduce with a **dedicated test workspace**, never a production
memory graph:

```bash
python services/hosted/scripts/live_smoke.py \
  --root /path/to/private-new-test-state --env-file /path/to/private.env \
  --graph YOUR_TEST_WORKSPACE_ALIAS --hermes-root /path/to/pinned-hermes \
  --hermes-python /path/to/hermes-python
```

The environment file needs only `DITTO_API_KEY`. The probe parses that one value
without executing the file. It creates a private local replay state with generated
Gnomon test tokens; protect and remove that directory when no longer needed. Reruns
reuse local idempotency keys. Omitting `--hermes-root` tests the MCP SDK client only.
The probe performs remote writes and is opt-in, not part of credential-free CI.

## Prospective example with independent agents

A second [recorded example](evidence/live-load-independent-agents.json) used real
Linux one-minute load averages, sampled prospectively on 2026-10-01. Two independent
Codex subagents used the actual pinned Hermes MCP client. This exercised agent
reasoning through Hermes transport, not a full Hermes LLM runner or Bittensor subnet.

Agent A compared historical-mean and last-value forecasts, selected **0.3701171875**,
and recorded its decision **141 seconds before** the target. The server restarted
before the outcome. A separate authorized process measured **0.23193359375**, first
reported it rounded to **0.2**, and recorded the full precision as a later revision.
The initial review and lesson were exported to the live Ditto test workspace.

After another restart, Agent B received only connection instructions and a search
query, without the expected answer. It retrieved the Ditto lesson, resolved its
ledger references, and independently reproduced the original MAE **0.1701171875**.
Its read-only write attempt was rejected; another project's credentials could not
resolve the evidence. After the precision revision and a third restart, it
recomputed MAE **0.13818359375**, checked `evidence_changed=true`, and verified that
the original lesson, actual and historical review remained unchanged. It correctly
classified the old lesson as historical rather than current numerical evidence.

The run exposed misleading write diagnostics: a hosted mutation could report zero
core writes before its enclosing transaction committed. The receipt now reports
core changes within that transaction; regression coverage checks a real write,
receipt replay and a duplicate observation under a new request key.

Clock regressions were also observed in this environment. The sampler discarded a
regressed history window. An early review correctly remained pending because its
cutoff preceded actual ingestion; a subsequent review used new cutoffs and a new
idempotency key. One recall rejected a freshly obtained cutoff as future and
succeeded after obtaining fresh server time. This reinforces the operational need
for a stable server clock; this example does not establish tolerance of clock skew.

Load average includes runnable and uninterruptible tasks, not CPU utilization.
Twelve overlapping samples and one outcome establish no forecasting advantage.
Agent B verified recorded evidence and arithmetic, not the source's real-world
truth or the narrative's causality. Private credentials and raw local test state
are excluded from the committed report.

## Automated local coverage

`tests/test_ledger_transactions.py` and `services/hosted/tests` cover:

- Atomic extension/core transactions, rollback and caught nested failure.
- Real Streamable HTTP and legacy SSE; SSE session ownership, revocation,
  cross-project denial, host/origin rejection and request size limits.
- Shared forecast/decision/actual/review/lesson reads across server restarts.
- Historical review immutability after actual revisions, and a clean-directory
  backup/restore with references preserved and old credentials revoked.
- Exact duplicate requests and conflicting keys; a failed receipt write rolling
  back the associated observation; restart recovery of unfinished requests.
- A real provider subprocess timeout and cleanup; durable unknown outcomes which
  never trigger automatic provider retries.
- Ditto delivery/recall against an explicitly labelled **local MCP contract peer**,
  including corrupted remote contents, lost save acknowledgements, explicit
  reconciliation, wrong-graph keys, and cancelled exports after project retirement.
- Actual pinned Hermes discovery and calls from separate processes when
  `GNOMON_HERMES_ROOT` is supplied. CI provides this checkout in its Hermes job.

The local peer tests are fault-injection coverage; they do not replace the live
Ditto result above. The core production regression suite is also run because the
new public transaction boundary touches existing ledger readers and writers.

## Container and concurrency

The Docker image built from the source wheels and passed a read-only-root,
unprivileged-user probe: initialize a named volume, forecast via real MCP, restart
the container, and resolve the same execution. Compose configuration validates.
Reproduce with:

```bash
docker build -f services/hosted/deployment/Dockerfile -t gnomon-hosted:test .
python services/hosted/scripts/container_smoke.py --image gnomon-hosted:test
python services/hosted/scripts/load_probe.py
```

Four independent local client processes completed **40 unique contended observation
writes and 40 exact retries**, with 40 records/receipts, complete distinct revision
numbers, and zero errors. This run took 3.010s including server/process startup;
median request latency was 69.4ms, p95 111.0ms and maximum 147.1ms. Every request
created a fresh MCP connection. These figures characterize this small local probe;
they are neither production capacity nor a user-count/SLA estimate.
The [machine-readable result](evidence/hosted-load.json) is retained.

## Limits of this evidence

- Hermes transport/registry compatibility is tested; no autonomous LLM learning or
  forecast-quality gain is established. Ditto narratives remain hypotheses.
- The live test used Ditto's MCP service; it did not exercise the Ditto web app's
  UI for adding Gnomon as a remote tool server.
- TLS configuration is supplied, but public DNS/certificate issuance was not tested.
- Tests cover restart and exception paths, not power loss, disk exhaustion or a
  production-scale database. There is only one supported service schema/backend.
- Source truth, model revision authenticity, managed provisioning, automatic
  adaptation, active-active hosting and contractual recovery targets are not claimed.
