# M5 development deployment bundle

`build_m5_ml_bundle` packages both frozen seed plans, worker capsules, worker
proofs, the fixed eight-series development jobs, original panel identity
metadata and the isolated host modules. It copies only explicitly identified
files, verifies their original hashes before and after copying, and verifies
both plans and launcher imports from the copied code directory. It does not
copy the source sales archive, reserved numerical targets, installed runtimes,
credentials or production ledgers. It makes no API calls or reservations.

The adapter also imports `m5_prepare.py`, which supplies its day-grid and
selection-prefix constants. The bundle pins this transitive dependency to the
hash in the earlier adapter/panel synthetic receipts. The live integration's
host source list and running worker remain unchanged.

The first prepared bundle is `results/m5-ml-bundle-prepare-001/build`: 101 files,
807,738 compressed bytes, SHA-256
`09e29271854ce089fbcb30f6eefe645460732c06b7c9c659d06ec661c885504f`.
Copied-input validation and isolated launcher help both passed. Independent
streaming verification checked all 102 archive members, including the inventory.
Supplying only the completed seed-7 pilot as a host proof rejected before
creating another bundle directory. All commands and failure output are retained.

This first artifact intentionally has no full-host proof and is not ready for
paid dispatch. Once both seeds' full integrations finish, build a fresh artifact
with `--host-preflight` pointing to the actual full receipt; the builder verifies
its source, plan, runtime and completion bindings. It never invents a passed
receipt from the pilot or bypasses the launcher's predecessor/runtime checks.

The candidate-100 terminal record still needs inspection and, if its original
audit fails as anticipated, explicit retained reconciliation. If admission code
changes for that reconciliation, preserve and verify the change separately;
do not relabel this older bundle as the new source. Production deployment and
runtime verification remain outstanding. No final data gate has opened and no
accuracy improvement is implied by successful packaging.

Receipt: `evidence/m5-ml-bundle-prepare-001.json`. Main and PyPI are unchanged.
