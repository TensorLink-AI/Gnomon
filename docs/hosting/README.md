# Shared Gnomon and Ditto: implementation plan

Status: milestone A design and offline storage validation on the `ditto` branch.
There is no hosted server, authorization implementation or production Ditto connector
in this change. The contracts below describe the intended hosted API, not new working
commands. The existing local Python, CLI and stdio MCP interfaces remain unchanged.

The acceptance scenario is:

> Hermes A records a forecast and decision, then exits. An authorized process records
> actuals. Gnomon reviews the forecast. A lesson is exported to Ditto. A fresh Ditto
> session verifies the evidence. Hermes B retrieves that experience.

This must survive client and server restarts. Shared forecasts, decisions, actuals,
reviews and lessons are the first-release scope. Self-hosted Gnomon remains useful
without Ditto. Automatic policy changes, managed customer provisioning, unified Ditto
login and additional database backends are deferred.

## Design records

- [Ownership, evidence and durable-reference contracts](contracts.md)
- [Current persistence audit and deployment envelope](persistence-audit.md)
- [Validation evidence and its limits](validation.md)
- [Machine-readable evidence reference](contracts/evidence-reference.schema.json)
- [Acceptance fixtures](contracts/acceptance-fixtures.json)

## Architecture boundary

The core owns numerical validation, scoring and temporal evidence rules. A separately
packaged service in this repository will own authentication, authorization, project
resolution, request lifecycle and deployment. The Ditto adapter will connect a selected
project graph to ledger evidence. The agent retains responsibility for proposing
experiments and explaining decisions. Saved prose remains hypothesis data.

Proposed structure, to be created in milestone B:

```text
src/gnomon/                         existing core package
services/hosted/
  pyproject.toml                   separate service dependencies
  src/gnomon_hosted/
    app.py
    mcp/
    auth/
    projects/
    persistence/
    integrations/ditto/
  tests/
  deployment/
docs/hosting/
```

The hosted package must use public core operations, not duplicate scoring or depend
permanently on private SQLite helpers. Core changes belong in the core and carry their
own regression tests. A local installation must not acquire hosting dependencies.

## Milestone A: contracts and storage validation

1. Inventory persistent records and transient handles on current remote main.
2. Define project ownership, permissions, durable references and failure semantics.
3. Exercise the current SQLite ledger through independent spawned processes.
4. Check concurrent revision allocation, retries, snapshot reads and clean restoration.
5. Record measured results with a bounded claim. Identify the gaps for milestone B.

This milestone is an offline feasibility gate. Permission fixtures are requirements,
not proof of an implemented access-control layer. Small synthetic probes do not certify
production throughput or recovery objectives.

## Milestone B: authenticated shared service

Implement one single-server deployment and one backend. Start with the SQLite candidate
only if the measured workload envelope remains appropriate. Add project-scoped, revocable
credentials and resolve every object within authorized project membership. Audit local
file loading, provider entrypoints and configuration before exposing any operation.

Expose only the operations needed for the handoff through the canonical Gnomon dispatch
layer. Use a maintained remote MCP implementation. Bind transient MCP state to its
principal/project; durable resource resolution must also work in a fresh session. Core
`data_ref`, `result_ref` and cache entries are not shared resources.

Implement durable operation receipts and a persisted review snapshot using core-produced
evidence. Couple ledger effects and receipts transactionally, or implement an explicit
recoverable operation marker. A service-control database commit followed by a ledger
commit is not atomic. Do not claim exactly-once execution across provider boundaries.

Gate: two independent clients share evidence across an actual server restart; a third
project cannot enumerate or access it. Inject failures before provider execution, after
provider completion, after ledger commit and before response delivery. Document each
result, including `outcome_unknown`.

## Milestone C: Ditto handoff

Validate Ditto's actual tool schemas, authentication, selected-graph behaviour, transport
and duplicate-save semantics. Configure a graph-scoped connection per project. A Ditto
credential is not automatically a Gnomon identity token. Keep separate credentials and
trust boundaries until an identity integration is agreed.

Export a compact lesson with durable references, provenance and checked facts separated
from narrative. Persist an outbox after the ledger operation; reconcile uncertain saves
before retrying. Search/fetch results are untrusted until project scope, evidence
eligibility and the referenced ledger have been checked. Historical evaluations use
frozen eligible memory; current semantic search is not a historical evidence snapshot.

Gate: run the complete acceptance scenario with real Hermes and Ditto connections. An
offline fake connector is useful for fault tests but cannot satisfy this gate. An outage
leaves exports pending or uncertain without undoing local evidence. Revised outcomes
produce explicit new versions; deletion disables pending exports before cleanup.

## Milestone D: self-hosted beta

Ship a versioned container, Compose configuration, persistent storage, TLS instructions,
credential lifecycle, health/readiness endpoints and structured logs. Add backup,
clean-machine restore, migrations and tested rollback boundaries. Export and import the
identity manifest with the evidence and artifacts; restore credentials separately.

Agree and measure concurrent-client, latency, database-size and recovery targets before
release. Test them against representative histories and quantile payloads, not just the
tiny milestone A fixtures. Installation must work without Ditto or Cloudflare.

## Milestone E: benefit and deployment decisions

Evaluate shared experience separately from automatic adaptation. Compare matched tasks
with/without shared evidence: handoff success, stale/incorrect claims, numerical
correctness, forecast quality, tokens, latency and cost. Freeze tasks and evidence
cutoffs. Inspect the current main router and reproduce any remaining defects rather
than assuming findings from an earlier working tree still apply.

Managed hosting, a second backend, unified identity and automatic model selection are
separate decisions after the service is measured. More API replicas are useful only
after the storage, job and session design supports them.

## External compatibility references

- [MCP Streamable HTTP and legacy SSE compatibility](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
- [MCP authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)
- [Ditto developer quickstart](https://heyditto.ai/docs/developer-quickstart)
- [Ditto inbound MCP connections](https://heyditto.ai/docs/mcp-connections/)
- [Ditto outbound memory MCP](https://heyditto.ai/docs/mcp-server/)

Ditto's published inbound guide specifies legacy SSE; its memory server documents
Streamable HTTP. These are different directions. Negotiate and test the versions
actually supported by each client before publishing a compatibility claim.
