# Rename impact inventory

**Nothing is renamed here.** The August 2026 design review noted a name
collision with AION (Zhan et al., arXiv:2605.25045) and asked for an
inventory so a human can decide. This is that inventory, measured
2026-08-02 at `pyproject.toml` version 0.4.0.

## Scale

137 files under version control mention the name (excluding `.venv`,
`results/`, `.git`).

## What a rename would have to change, by blast radius

### 1. Published identifiers — breaking, coordinated with users

| Surface | Current | Notes |
| --- | --- | --- |
| PyPI distribution | `aion-forecast` | v0.4.0 tagged; a rename means a new project, and the old name must stay as a redirect shim or installs break |
| Import package | `aion` (`src/aion/`) | every user's `from aion import forecast` |
| Console script | `aion` (`[project.scripts]`) | every documented command, every user's shell history and scripts |
| MCP tool prefix | `aion_*` — 24 tools (`aion_forecast`, `aion_detect_anomalies`, `aion_route`, …) | **frozen by `COMPATIBILITY.md`**; renaming breaks the v0.2 contract outright |
| GHCR image | `ghcr.io/tensorlink-ai/aion` | tags `0.4.0`, `0.4` published |
| Hermes plugin | root `__init__.py`, `plugin.yaml` | `hermes plugins install <repo-url>` resolves by repo root |
| Store/artifact strings | `store:<dataset>` refs, `forecast_<hex>` ids | ids are content-addressed and salted with the version, not the name — unaffected |

### 2. Internal-only — mechanical

- `src/aion/**` module paths and their imports (~40 modules).
- `tests/**`, `benchmarks/**` imports; `pyproject.toml` `pythonpath`,
  `testpaths`.
- `skills/forecasting/SKILL.md`, `integrations/hermes/SKILL.md` (kept
  byte-identical by a sync test — rename both or the test fails).
- Docs: `README.md`, `docs/*.md`, `CHANGELOG.md`, `COMPATIBILITY.md`.

### 3. Not renameable without a deprecation cycle

The MCP tool names are in the frozen v0.2 set. Any rename there is a
contract break, not a rename — it would need both names served for at
least one release, with the old set marked deprecated in
`COMPATIBILITY.md`.

## The cheapest option that addresses the collision

Keep every identifier and disambiguate in prose: a "Relation to prior
work" section in `README.md` (added 2026-08-02) stating what this project
is and is not relative to AION and TimeClaw. Identifiers stay stable;
search-engine and reader confusion is addressed where it actually
occurs.
