# Amendments before any live outcomes

2026-09-15, A1 (implementation review): the existing aggressive repair resolves
conflicting duplicates by **last row in file order**, not their arithmetic mean
(src/gnomon/repair.py). Eval 2's fixed policy and oracle use last-in-file. This
corrects the prospective protocol before corpus construction or model execution.

2026-09-15, A2 (release lookup): fetched origin/main, commit
59a6d81709a4625bf042e7ca152aa5f12534c28a, which also declares 1.1.9. No remote
v1.2.0/1.2.0 tag was returned. User explicitly requires work on top of 1.2.0.
Harness development may continue, but confirmatory execution remains blocked on
identifying that release. Do not bump a version string to manufacture a release.

2026-09-15, A3 (repeat identity): use the existing driver with an additive optional
public-input flag, `hide_case_id_from_model`, in available_at_cutoff. The shared
loop removes only the journal identifier and this flag from model-visible input.
It preserves original case IDs for submissions, journals and grading. All arms
use identical behavior; this is essential to genuinely identical repeat prompts.
The driver passes that same view to tool backends, so their public case.json cannot
reintroduce differing repeat IDs. Normal cases without the flag retain their IDs.

2026-09-15, A4 (executable grading specification): require the named repair codes
in final disclosures as well as facts.repair_operations, so automated omission
grading does not depend on another LLM. Preserve prose and label this a disclosure
format requirement, not evidence of a silent physical data mutation. Reference
row comparisons normalize timestamps and tolerate 1e-8 scale numerical noise.
Use a two-sided source-window cluster sign-flip test (10000 draws, seed 20260915,
plus-one Monte Carlo correction), then Holm across three primary comparisons.
This assumes exchangeability under the null; sequential arm passes/provider drift
can violate it. Confidence intervals remain the registered stratified bootstrap.

2026-09-15, A5 (audit retention): all arms request bounded tool-result retention in
the existing loop, at most 128 KiB/task in addition to existing arguments/digests.
Overflow is explicitly marked; absent receipts are unaudited, not evidence of no
repair or no leakage. Public local task data only; no credential/header capture.
The retention flag is removed from model-visible input in every arm.

2026-09-15, A6 (offline reference validation, no agent outcomes): the first contract
test found monthly interpolation uses calendar-grid step position, not elapsed
seconds. Fixed-policy task instructions and reference now explicitly use grid
position, matching the runtime. Initial offline test batch: 81 passed, one failed
reference check. Preserve this finding rather than presenting the initial oracle
as independently validated. Recheck against actual 1.2.0 source before launch.

2026-09-16, A7 (remote migration before live outcomes): user authorized execution
on an existing Gnomon Targon CPU pod. The previously missed pinned wheel was found
at /root/gnomon-ledger-ml-v3/assets on wrk-kadzj08j3t1o. Wheel SHA256 is
030a5cd063c424482bebdc2a522aa98a8f4bf88285bab31c4e47ddc9afa04f2f;
embedded clean build is 1.2.0+ga38cd0cad353.s9723394ccb6d, source commit
a38cd0cad35383e5f10021abf3aa20d4c16923be. Verify extracted Python source hash before
using it. This resolves the missing runtime artifact; earlier local checks remain
1.1.9 checks and are not relabelled. Select wrk-tjdrfztojlsn (gnomon-arena-roi-117),
with 4 allocated CPUs and approximately 50 GB memory limit, in a fresh directory.
Its /proc/meminfo reports the physical host, so the guard now uses the smaller of
host available memory and memory.max minus memory.current. All existing thresholds
remain unchanged; cache is treated conservatively as charged memory. No efficacy
results preceded this correction or migration.
