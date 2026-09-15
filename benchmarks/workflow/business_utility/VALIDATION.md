# Development validation, 2026-09-15

**Zero live agent tasks have run. Business-outcome counts and treatment effects
are not measured.** This is harness validation, not evidence of Gnomon uplift.

Pre-registration: commit `798ae907`, before implementation checks or model outcomes.
The targeted test command in README completed with **85 passed in 9.36 seconds**.
This includes existing matched-runner and scripted-driver tests, not 85 independent
business scenarios. The first batch found an incorrect monthly interpolation
reference; [A6](AMENDMENTS.md) records the correction before live execution.

Checks exercised all 80 generated portable snapshots against actual snapshot
reads, all eight messy files against actual repair behavior, repeat-prompt identity,
threshold arithmetic/refusals, original-answer preservation, missing-answer
accounting and matched configuration preparation. A small process fixture verified
that resource-stop cleanup reaches an owned child in a separate session and leaves
an unrelated process running. No 1.2.0 container smoke or live guard trial occurred.

Development runtime declared `1.1.9`, with pre-existing local modifications.
Runtime source fingerprint at validation:
`e994a3fd7334aacbca4bfbe772e141170051a739ea69c38f5e04f91bbacc1ac8`.
Fetched origin/main `59a6d81709a4625bf042e7ca152aa5f12534c28a` also declares 1.1.9.
This cannot be represented as a replication on 1.2.0.

Development corpus, manifest, machine-readable validation and three pending report
files are under `results/business-utility-development-20260915/`. They are local
development artifacts, not a frozen confirmatory corpus or published results.

Remaining launch gates:

- Identify the exact 1.2.0 source ref; apply this harness on top without importing
  unrelated working-tree changes. Revalidate and rebuild its existing service image.
- Select the model transport/account and per-arm allocation. The existing workflow
  driver does not currently implement the native Codex account transport used by
  the separate historical Ditto experiment.
- Freeze and commit the 1.2.0 corpus, amendments, source and matched configuration.
- Run the resource-limited smoke check, then execute the registered matched trials
  in order. Keep failures and stopped tasks in the full denominator.
- Audit execution receipts and publish measured reports, including null/adverse
  results and all registered claim limitations.
