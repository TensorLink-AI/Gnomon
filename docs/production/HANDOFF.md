# Delivery checkpoint — iteration 29

Updated 2026-09-07 on `codex/v1.0.0-clean-surface`, based on merged `main`
commit `3d2835f`.

## Current scope

Gnomon 1.0 has one provider-neutral Python/CLI/MCP execution contract, optional
budgeted evaluation, an optional current-format temporal store and ledger, and
optional explicit date/time calculations. User-selected callables and Ephemeris
remain the provider boundaries. The agent skill ships with the wheel.

Pre-1.0 artifact and TrackingStore imports, compatibility documentation, database
schema upgrades and the MCP profile selector are removed. Fresh temporal stores
and ledgers carry explicit database identities; incompatible files fail without
mutation. Revision-aware observations and current ledger operations remain because
they serve present data correctness, not compatibility.

## Verification

- Full local production and benchmark suite: 856 passed, 29 opt-in skips.
- Focused ledger-history regression: 62 passed.
- Focused agent/docs/session/schema checks passed.
- Ruff, compilation, whitespace and weighted-progress validation passed.
- Fresh `1.0.0` wheel/sdist and separately packaged provider example built.
- Twine metadata and clean offline installed-wheel Python/CLI/MCP/provider/ledger/
  plugin journeys passed; installed runtime reported `1.0.0`.
- Wheel SHA-256: `c69dff7db6eb6eae8d725b9ce5a3ec753e8fd74b691683ddeb5c177508aecc3d`.
- Source SHA-256: `35abecd10e1749d369b54fa7d7c361b802661b12582f0ab9f888f12efe3d831b`.

## Release state

The change is not yet merged or tagged. Push this branch, open a PR to `main`,
wait for all CI and container checks, merge the accepted commit, then create
`v1.0.0` at the resulting `main` commit. The tag-triggered release workflow must
build again, create the GitHub release and publish through PyPI Trusted Publishing.
Verify the PyPI JSON record and exact wheel/sdist names after it succeeds.

Do not stage or remove unrelated root scratch files (`-`, `Continue`, `Current`,
`Immediate`, `Use`, `accelerate`, `actual`, `cases.`, `optimizing`, `that`).

## Remaining evidence limits

The delivery score remains 97/100. Live Ephemeris verification and an actual
matched agent comparison are still separate evidence gates. Version 1.0 stabilizes
the package interface; it does not assert forecast superiority, calibrated model
uncertainty or action authorization.
