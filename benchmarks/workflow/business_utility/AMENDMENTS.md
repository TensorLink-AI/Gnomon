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

2026-09-16, A8 (confirmed deployment and model, before trial outcomes): user
confirmed gnomon-arena = wrk-kadzj08j3t1o, distinct from the ROI deployment.
Use a fresh workspace on that original pod. Select the existing DeepSeek model
configuration, deepseek-v4.1-flash via https://api.engy.ai/v1, for every arm.
Immutable model weights/revision are unavailable and remain null. Registered
generation/task/token/tool/round/retry limits are unchanged. Freeze a $5 reported
service-charge stop per arm per evaluation ($45 across nine allocations); this is
not a hard provider/account cap. One excluded task-free readiness request returned
HTTP 200, 8 prompt and 4 completion tokens, and x_engy.charged_micro=1. The common
transport now reads that integer charge using Engy's documented micro-USD monetary
convention (https://engy.ai/docs/agent-api); absence/invalid values stay unknown.
It accepts this field only for the explicitly configured Engy origin, never
estimates prices from tokens, and preserves existing usage.cost when supplied.
This is provider-reported accounting, not an independently verified invoice.
No budget increase, model substitution or failed-task retry occurs automatically.

The image stream between pods was interrupted before completion. Rebuild from the
same existing pinned Dockerfiles/requirements and verified wheel on gnomon-arena;
pin the resulting actual image IDs before dispatch. Local files are never used as
an alternative execution host. The initial missing pyproject preflight log and
interrupted transport remain infrastructure history, not agent task failures.

2026-09-16, A9 (preflight fixture timing, before trial outcomes): remote container
integration checks passed 52 tests and failed one because Docker startup exceeded
the lifecycle fixture's three-second command timeout. Increase that test's lifetime
to ten seconds and observation window to fourteen seconds, preserving its check
that PID 1 terminates without host cleanup. This changes no live task limit;
the registered 180-second task budget remains unchanged. Retain the initial log.

2026-09-16, A10 (outcome-informed correction, user requested): stop trial-001 and
retain all observations, skipped/error rows and partial passes. Eval1 lean answered
0/80 (80 token stops); ordinary 14/80. Eval2 lean had one outer timeout followed by
127 spending_usage_unmeasured rows, not 128 independent model failures. No uplift
claim is supported. This follow-up reuses exposed cases and is exploratory;
a new unseen corpus is required for subsequent confirmatory evidence.

For the follow-up all arms receive file paths, sizes and SHA256 hashes in the
initial prompt rather than duplicated file bodies. Full identical files remain
materialized in every backend and accessible with Python. Increase cumulative
reported token allowance from 32,000 to 128,000 equally across arms; retain 12
rounds/tools, 2048 output tokens, temperature zero, model, cost limits and 180-second
outer task deadline. The driver reserves min(15 seconds, 10% of outer deadline)
for cleanup/accounting; agent/backend work gets the remaining 165 seconds here.
Unknown spending still blocks further tasks; never fabricate zero charges.
Freeze corrected code/config before follow-up outcomes. Run an explicitly labelled
one-case-per-evaluation-per-arm diagnostic subset first, with no automatic retries.
Do not merge its results with trial-001 or count it as confirmatory evidence.

2026-09-16, A11 (decision wire format, after A10 diagnostic): all nine A10
answers completed, but all three Eval1 answers used choices.plan="plan".
Explicitly document in every Eval1 prompt that choices.plan must be "approve"
when reported_nmae <=1.0 and "review" otherwise, and that "plan" is a field
name rather than an answer value. All arms receive the identical clarification.
No parser coercion, answer rewriting, extra retry or grader relaxation is added.
Original invalid answers remain invalid. All other A10 settings stay fixed.

Before a larger exploratory follow-up, run the first clean/trap pair in Eval1
across all three arms (six tasks). This exposed-case diagnostic tests completion
and decision format, not uplift. Freeze before calls. Launch the full corrected
exploratory sweep only if all six submit auditable decisions; otherwise retain
failures and investigate. The full sweep reuses exposed cases and cannot provide
fresh confirmatory evidence. Keep all previous runs separate.

2026-09-16, A12 (launch-only correction): all six A11 diagnostic decisions passed.
The supervisor then failed before the full dispatcher started because full.log
already held the full-arm diagnostic log. Use full-dispatch.log for the dispatcher.
Preserve diagnostic logs and decisions. Start the frozen full-followup configuration
once, directly through its existing dispatcher, after committing this correction;
do not replay the gate. No prompt, budget, model or grading changes.

2026-09-16, A13 (user-requested move to ROI): stop the original arena full
follow-up and preserve its observations/journal as an interrupted trial. Restart
as a separate exploratory run on wrk-tjdrfztojlsn (gnomon-arena-roi), using its
verified prebuilt image IDs. The existing resume CLI would retry failed tasks;
do not use it or merge the interrupted trial into the new comparison. Repeated
cases remain exposed and all previous failures remain in their original records.
Pin the actual destination images, code, environment and identical arm settings
before new calls. Keep one worker and unchanged memory, task and cost limits.

2026-09-16, A14 (diagnosis; no full restart): ROI run stopped for investigation.
Seven first arm failures were OpenRouterError; five immediately followed a
JSONDecodeError on submit_answer. The driver discarded transport details.
An excluded, pre-recorded two-request synthetic probe returned HTTP200 for a
valid prior tool call and HTTP400 for malformed prior arguments; the provider
explicitly required valid JSON in assistant tool-call history. Two other historic
transport errors cannot be identified from the retained class-only logs.

Do not resend malformed tool-call arguments as conversation history. End that
task with invalid_tool_arguments (or model_output_truncated when finish_reason is
length), retaining measured usage and failure in the denominator. No coercion or
extra model request. Retain safe transport error codes/status without raw provider
bodies; stop dispatch of subsequent arms if accounting is incomplete. Existing
unknown spending remains unknown. No output/token/budget/model increase. These
changes require a new frozen experiment before any subsequent scored run; the
current run and its source remain intact. No full restart is authorized by this
amendment itself. The diagnostic probes are excluded from efficacy denominators.

2026-09-16, A15 (user requested completion of the fix): verify that a malformed
model answer is retained with its measured cost and that the existing runner can
execute subsequent tasks without an unknown-spend cascade. Then run a separately
frozen nine-task live diagnostic, all three arms on these exposed cases:
e1-pedestrian_counts_daily-1-clean;
e2-pedestrian_counts_daily-0-autonomous-5;
e3-pedestrian_counts_daily-4-below.
These include previously failing locations, chosen before new outcomes. Keep
model, budget, arm order, corpus content and scoring fixed except the A14 loop
and diagnostics changes. No retries, no selecting alternate cases after outcomes.
Success for the transport fix is complete measured accounting for all diagnostic
tasks, with no malformed-history replay or unknown-spend cascade. Report answer
completion separately; a retained malformed answer is still a task failure.
A genuine API/network failure must stop the diagnostic and keep charges unknown.
Do not automatically start another full sweep. All diagnostic evidence remains
exploratory and separate from the interrupted trial.

A15 reporting correction: after six live tasks all retained complete accounting,
but the report crashed because the diagnostic's autonomous-only Eval2 subset has
no fixed-policy primary comparison. Leave the missing effect null and omit that
comparison from familywise testing. Fix reporting separately, retain all six
observations, and run only the three still-unattempted Eval3 tasks with the exact
original frozen agent code/configuration. No repeated model tasks or changes to
grading criteria. Preserve the original supervisor exception as infrastructure
history. This is a diagnostic subset edge case, not a measured effect.

A16 (user-requested malformed-answer recovery): enable at most one model-driven
format correction per task, equally across arms. Applies only to a sole malformed
submit_answer call, never mixed or executable tool batches. Preserve the original
arguments in the bounded audit trace (with hash/omission marker if too large),
quote the original invalid output as assistant text in subsequent model context,
and ask for a valid JSON submission. Never send invalid tool-call history, invent
corrected values, or turn it into a host-generated answer. All original time,
token, round, tool and charge limits remain in force. No correction if budget or
usage is exhausted/unknown. A second malformed answer remains a failed task.
A recovered answer is identified explicitly by answer_format_recoveries; do not
claim first-attempt success. Default off for other workflow experiments; enabled
in the business-utility corpus. Prior runs remain unchanged.

Validate with scripted regression cases and an excluded synthetic live transport
check using the exact corrected message representation. That check starts from
a supplied malformed string and tests whether the provider accepts the recovery
history; it is not a spontaneous model failure, scored task or uplift experiment.
No automatic full evaluation restart is part of this amendment.
