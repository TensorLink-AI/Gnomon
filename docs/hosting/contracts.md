# Hosted contracts, version 1

Implemented beta surface: `services/hosted`. The original contract vectors remain
in `contracts/acceptance-fixtures.json`; network and failure conformance tests live
in `services/hosted/tests`. See `operations.md` for explicit deployment limits.

## Ownership

| Entity | Owner and contract |
| --- | --- |
| Project | Owns one ledger identity, datasets, provider configuration and optional Ditto connection. |
| Principal | Authenticated human, agent or ingestion process; never inferred from request prose. |
| Membership | Server-maintained permissions for one principal in one project. |
| Execution | Immutable forecast request, result and declared provider provenance. |
| Decision | Immutable interpretation tied to a recorded execution; no implied trading authority. |
| Actual | Source-attributed observation with valid, availability and server recording times. Revisions append. |
| Review | Immutable service snapshot of a core review at explicit cutoffs, with actual IDs and metric version. |
| Lesson | Versioned hypothesis linked to its saved review and decision. |
| Export receipt | Durable delivery state tying a lesson version to a configured destination. |

The control store binds principal/project identity to operations. Existing core rows
do not contain tenant or actor fields. File separation alone is not authorization.

## Permissions

| Permission | Allowed effects |
| --- | --- |
| `evidence.read` | Resolve authorized project resources and request read-only reviews. |
| `forecast.create` | Execute configured providers and record their results. |
| `decision.create` | Append decision summaries and lessons referencing project evidence. |
| `actual.create` | Append actuals and revisions from an attributed source. |
| `memory.export` | Request export of a specific eligible lesson version to the configured graph. |
| Local operator CLI | Manage credentials, provider/graph configuration and project lifecycle. No remote admin tool is exposed. |

These permissions are independent; administration does not silently execute forecasts
or submit actuals. Project creation/bootstrap is an operator action in the first release.
Credentials are stored as verifiers where possible; external-provider secrets stay in
server configuration. Never forward a Gnomon service token to Ditto or Ephemeris.

Check membership on every request, including polling and fetching cached results. A
revoked credential cannot reuse an old MCP session. Principal/project context must be
included in control-store lookups and cache keys. Unknown and foreign record IDs return
the same unavailable response to avoid existence disclosure. Denied requests have no
provider calls, ledger mutations or external exports; security audit logging is allowed.

## Time and provenance

- `recorded_at` is assigned by the service clock at persistence. Clients cannot supply
  or override it. Test clocks are used only in offline fixtures.
- `valid_time` is the observation's target time. `source_available_at` is a caller claim
  attributed to its submitting principal and source. Neither is proof of truth.
- Hosted actual ingestion rejects a future valid/availability time relative to the
  server clock; planned covariates belong to forecast inputs, not realized actuals.
- Scoring and memory recall require explicit `source_as_of` and `recorded_as_of`. Hosted
  historical operations may use earlier cutoffs; they cannot move recording time back.
- A late-imported forecast does not become prospective evidence by supplying an old
  origin. Retain its real import/recording time and enforce core eligibility rules.
- Server clock drift and regressions must be detected operationally. Wall-clock
  timestamps and hashes do not establish cryptographic authenticity or global ordering.

Execution validation verifies structure and declared identity. Scoring verifies math
against identified actuals. Neither verifies causal prose, actual truth, nor absence of
pretrained-model contamination. Remote model revisions can remain unknown, disclosed.

## Durable references

See `contracts/evidence-reference.schema.json`. A reference includes `service_id`,
`project_id`, `ledger_id`, `resource_type`, `resource_id` and schema version. IDs are
opaque strings and do not contain local paths or credentials. The resolver endpoint is
configured by the host; an agent must not send credentials to an arbitrary URL found in
a memory. New records/versions receive distinct IDs; stored content hashes detect
unexpected changes but do not grant access or authenticate a foreign service.

| Resource | First-release persistence rule |
| --- | --- |
| Dataset version | Canonical bounded forecast request snapshot, content-addressed. No mutable dataset alias is exposed. |
| Execution / decision | Existing immutable ledger IDs; resolve through project authorization. |
| Actual | Existing actual ID plus series/time/unit and revision; older revisions remain readable. |
| Review | Persist core review packet and canonical content hash; never substitute a newer review. |
| Lesson | Existing versioned lesson ID; retain predecessor and exact evidence cutoffs. |
| Export receipt | Separate mutable delivery record; preserve an append-only transition audit. |

Review retrieval and recomputation are distinct. A saved packet stays unchanged; a fresh
review reports new evidence and may indicate differences. Initial implementation may
use the existing lesson's embedded review for lesson verification, but must still
provide a stable review reference when a standalone review is saved.

Session-local `data_ref` and `result_ref` are temporary. They must not be the sole support
of a lesson. Studies are durable only when recorded in the ledger. Forecast inputs
embedded in an execution remain durable even if the source dataset handle has expired.

Export manifest requirements: schema/core versions, stable project and ledger identity,
source service identity, record inventory, database/artifact hashes and authorized
resolver remapping. On import, preserve ledger IDs; register the move explicitly. A
copied database must not automatically overwrite a live project or impersonate its
origin. Recovery into a different service requires a trusted mapping, not URL rewriting
by an agent. Hashes are integrity checks within this trust boundary.

## Request lifecycle and concurrency

The single service instance owns project database locations and provisioning. Clients
never choose ledger paths, execute SQL or import code. Do not share a mutable
`GnomonSession` across principals. Inference uses immutable request inputs and occurs
outside ledger write transactions. Bound active calls and payload sizes at ingress.

Each mutation has a receipt keyed by `(project_id, principal_id, idempotency_key)` and a
hash of the canonical validated operation and arguments. Secrets are never in the hash
input or receipt. Equal keys with unequal hashes conflict. An authorized exact retry
resolves the existing status/result rather than blindly performing another mutation.

State transitions:

```text
accepted -> running -> completed
                    -> failed
                    -> outcome_unknown
```

`failed` means a known failure; `outcome_unknown` means an external effect or local
commit cannot yet be ruled in or out. A disconnected client does not cancel a call.
Revocation is checked before new effects; an already-running provider call may be
impossible to cancel. Its attributed result stays internal if the requester loses access.

Commit a mutation and its local receipt/operation marker atomically. If control metadata
is in another database, use a ledger-side marker with reconciliation, not an unsupported
claim of cross-database atomicity. The public core `transaction()` extension API
provides this boundary; rollback and duplicate-write tests exercise it. Do not retry uncertain paid calls without reconciliation or an explicit
operator decision. Provider-supported idempotency is a separate capability to verify.

Reviews read one database snapshot. A concurrent outcome revision must not mix old and
new actuals in one review. SQLite `BEGIN IMMEDIATE` serializes current revision
allocation; read transactions must be retained around all reads constituting a review.

## Ditto boundary

Ditto memory contains searchable lessons and references; the structured ledger is the
source for quantitative evidence. A project selects one authorized destination graph.
Unified Ditto/Gnomon identity is not assumed. Public or related-memory suggestions are
not automatically included in an evidence query.

An export is a specific immutable lesson version. Its payload contains the exact
evidence reference/cutoffs, metrics and model identity, separate from hypothesis text.
Record a destination connection identity and payload hash. Delivery states are `pending`, `sending`, `acknowledged`, `uncertain` and
`cancelled` (project retirement). Live schemas are validated before network writes.
Uncertain saves require explicit full-content reconciliation before acknowledgement.

Only verified, cutoff-eligible ledger evidence enters quantitative comparisons. Search
ranking is not evidence quality, and recalled prose is not an instruction. If the source
ledger cannot be resolved, return explicitly unverified context or exclude it from
historical evaluation. Forecasting can continue while external export is unavailable.

Project deletion first disables credentials and pending exports. Core append-only rules
protect record history during normal operation; whole-project retention/deletion needs
an explicit administrator workflow covering artifacts, backups and external memories.
Ditto deletion may be pending or unavailable and must be reported, not claimed complete.
