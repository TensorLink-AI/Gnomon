# Rename impact inventory

**The rename happened.** This file was written on 2026-08-02 as an
inventory so a human could decide whether to rename. It recommended
*keeping* the name and disambiguating in prose. That recommendation was
overruled: `Aion` became `Gnomon` in **v0.5.0**, repository included.

The trigger was the name collision with AION (Zhan et al.,
arXiv:2605.25045), noted by the August 2026 design review.

## What actually changed

| Surface | Before | After |
| --- | --- | --- |
| Distribution | `aion-forecast` | `gnomon-forecast` — `gnomon` alone is taken on PyPI |
| Import package | `aion` (`src/aion/`) | `gnomon` (`src/gnomon/`) |
| Console script | `aion` | `gnomon` |
| MCP tools | `aion_*` — 22 tools | `gnomon_*` — same 22, same inputs, same envelope |
| Environment | `AION_*` | `GNOMON_*` |
| Default output | `aion-output/` | `gnomon-output/` |
| Config file | `aion.yaml` | `gnomon.yaml` |
| Repository | `TensorLink-AI/Aion` | `TensorLink-AI/Gnomon` (GitHub redirects the old URL) |
| GHCR image | `ghcr.io/tensorlink-ai/aion` | `ghcr.io/tensorlink-ai/gnomon` |
| Benchmark records | `aionbench.jsonl` | `gnomonbench.jsonl` |

Deliberate exception: **`AION` still appears** in `README.md` and
`docs/integration-plan-review-2026-08.md`. Those are citations of Zhan et
al.'s system — the one this project renamed *away from*. They must never
be substituted, and the doc-drift suite has no opinion about them, so
check by hand if you ever re-run a bulk rename.

## What it did not cost

The two expensive risks this inventory originally flagged did not
materialise:

- **No PyPI break.** `aion-forecast` was never actually published, so
  there was no installed base to redirect and no shim to maintain. The
  inventory assumed otherwise.
- **The name never entered a forecast ID.** Content-addressed
  `forecast_<hex>` ids are salted with the runtime *version*, not the
  name. The ids in `tests/goldens/` did move — but only because v0.5.0
  bumped the version, which every release does.

The one real cost was the frozen v0.2 tool contract: renaming `aion_*` to
`gnomon_*` is a break, not a rename. It was taken as a hard break with no
alias period, on the grounds that serving `aion_*` as a deprecated alias
would have kept the very name the rename existed to remove. See
`COMPATIBILITY.md`.

## The original inventory (2026-08-02, superseded)

Measured at `pyproject.toml` version 0.4.0. 137 files under version
control mentioned the name, excluding `.venv`, `results/`, and `.git`.
The recommendation was to keep every identifier and disambiguate in a
"Relation to prior work" section of `README.md`, on the grounds that
identifier stability was worth more than search clarity.

The counter-argument that won: a name you cannot claim on PyPI and cannot
use in a citation without a disambiguating footnote is not a stable
identifier. Every further release under it would have raised the cost of
the change that was going to happen anyway.
