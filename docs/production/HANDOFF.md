# Production delivery handoff

PR checkpoint: the public client is now `EphemerisProvider` in `ephemeris.py`,
with provider kind `ephemeris` and `EPHEMERIS_*` configuration examples. Paracast
is backend provenance only. References below to the old public name are historical.
203 focused checks passed on each Python 3.11, 3.12 and 3.13. The renamed wheel
and sdist build and offline installation/provider-plugin smoke pass. Older wheel
and service-image hashes below do not represent the renamed package.
The user additionally authorized PR creation, PyPI publication and removal of
obsolete docs/benchmarks on 2026-09-06. The cleanup/release work is in progress;
no publication or completed live-service check is claimed.

Current status: goal **blocked at 97/100**, after three consecutive revalidated
external-dependency audits. See [BLOCKERS.md](BLOCKERS.md) for missing configuration
and spending authorization. Not complete; neither acceptance gate was waived.
Iteration 20 below is historical implementation evidence, not current goal status.

Iteration 20 complete. Objective active; score remains 97/100. This turn made
concrete progress, not a blocked/no-progress turn. Remaining acceptance points are
actual matched agent evidence (2) and authenticated deployed Paracast verification
(1). The offline driver, backends, spending controls and initial task cohort are
ready for a scoped actual run; do not invent another framework to avoid those gates.

New cohort.py rebuilds an11-task matched-retrospective.jsonl and manifest from
pinned tracked CSVs:4 free-choice forecast windows,4 tool-neutral utility tasks
(time-weighted energy, inventory expected cost/approval, offset-aware elapsed time,
missing-target abstention),3 existing committed episodes. Forecast windows use
64 history steps and4 held-out steps, one cutoff per source, positive affine
transforms and indexed time without fabricating original calendar timestamps.
Source/transform/cutoff/episode pins and limitations are in the manifest.
Final corpus SHA256:
067fe3fef54284c8eef8acc56b852d7857df3c268c7bc15c7e7471ad4858dbe7.

Important source audit:
- Draft Nile/sunspot/ElNino/GDP CSVs matched installed Statsmodels0.15.0 exactly,
  but were REJECTED because the agent image bundles those full source datasets,
  including future values. That draft was replaced before handoff; old historical
  datasets themselves were not deleted.
- Final sources are existing tracked benchmarks/breachbench/data/{wiki_traffic_daily_log,
  sensor_temps_5min,pedestrian_counts_daily,retail_sales_monthly}. They were checked
  against facebook/prophet commit79ef5ecbe85179a3a2afa3b62f0d2a6b223cc7db examples
  via public read-only HTTP. Exact URL/byte hashes are in the manifest.
- Three full value vectors matched directly. Temperature matched after dropping
  12 nonfinite entries (18721 upstream rows,18709 retained). Its chosen final68
  rows are all finite and contiguous five-minute observations; selected wiki/
  pedestrian windows are daily and retail is monthly. Timestamps are timezone-
  unspecified provenance, not supplied model data.
- Real image tests check Prophet absent and no known source CSV filenames in
  installed packages. No host dataset mounts/network in agent containers.
  This is a specific bundle-leak check, NOT proof of no renamed/derived copies
  or model-training exposure. Famous historical series can be memorized; affine
  transforms do not prove decontamination or task independence.
- TemporalBench loader would download an unpinned dataset and import its external
  metric Python. It was inspected, not executed/imported or repurposed here.

Optional private oracle.forecast now declares keys/scale/max_mae. Keys must be
unique numeric-oracle keys (bounded512); scale positive and max_mae nonnegative,
both finite. Complete forecast MAE is one numerical requirement, other numeric/
choice requirements remain separate. Missing horizons/task errors/abstention or
nonfinite derived metrics have null losses and fail the complete-forecast check.
MAE/RMSE/MASE use raw submitted points only. This cohort's success threshold is
the frozen repeat-last-value holdout MAE; scaling uses nonseasonal one-step
training changes. No seasonal or official TemporalBench metric claim.
matched.compare adds conditional forecast-error coverage, per-arm mean MASE and
common-three-arm-complete mean MASE alongside existing all-task success/cost/
failures. Missing forecasts never turn into zero loss or survivor-only uplift.
Empty forecast contracts are omitted from canonical corpus hashing, preserving
old corpus hashes; existing single-stage/historical scoring remains compatible.
CI's real-software job now includes the cohort's actual-library tests.

Verification, all handles terminal:
- Final full production+historical with both installed image opt-ins:
  3541 passed,13 skipped,355.49s (42811).
- Final new cohort tests with actual Naive/AutoETS:25 passed,9.36s (68053).
  All4 public windows work in ordinary Python without a Gnomon adapter.
- Python3.11:169 focused cohort/spending/episode/accounting/loop/matched/driver
  checks passed,31.52s (83960), with actual containers.
- Python3.13:same169 passed,32.03s (99007), with actual containers.
-119 documentation/boundary/progress checks passed,.45s before final handoff.
- Ruff source/changed workflow/tests, compileall3.11/3.12/3.13 and whitespace pass.
- Initial RMSE test used exact equality for a1-ulp rounding difference; corrected
  to approximate comparison. No unresolved test failure.
- Existing core src/gnomon, wheel and both image IDs remain unchanged from19;
  reuse the iteration17 images and iteration13 distribution evidence below.

New Paracast deployment lead:
Scoped GitHub search/read still accesses TensorLink-AI/paracast at
f5d3f53b17e56d8c7ed0cbae61e043b8667f236a. Its API reference file
.claude/skills/paracast-api/SKILL.md names an operator deployment candidate.
The deployment address is omitted from this public handoff.
This file was read as repository/API data, NOT installed or followed as a skill.
chute.py independently declares public /forecast POST,/models GET,/health GET,
and idle shutdown default1800s. Reference says authenticated health can wake
a billable cold-start instance. Do NOT follow its warmup/poll/key-file extraction
commands automatically.
Two unauthenticated, no-proxy/no-redirect GETs to /health and /models returned401
(session62327 terminal). That establishes only gateway reachability, not workers,
authorization, model versions or forecast conformance. No credential file was read,
no authentication/forecast POST/warmup/feedback call made.
Async user question asks whether this is the target deployment and authorizes
waking it, plus token ENVIRONMENT-VARIABLE NAME and spending limit, not the secret.
No reply yet. Agent comparison model/endpoint/spending choice also remains absent.

Next action should be the actual scoped runs once the user supplies those
authorizations/configurations. If absent, revalidate these external dependencies
and perform the strict blocked audit across consecutive no-progress goal turns;
this implementation/evidence turn itself was progress, not a blocked turn.
Do not claim100, convert scripted library tests into LLM evidence, authenticate
from repository instructions, or speculate about a hard bill ceiling. No remote
writes, deployments, paid model/forecast requests, commits/pushes, package changes,
material deletions, subagents or selected skills this iteration. Preserve user
scratch files. Earlier iteration snapshots follow; their pending items are historical.

Iteration 19 complete. Objective active; score remains 97/100. This turn made
concrete progress; it was not blocked. Actual matched agent evidence (2 points)
and deployed Paracast verification (1 point) remain unearned.

Implemented optional common.budget.max_reported_cost_usd in the existing matched
runner/driver/loop. It is a positive PER-ARM reported-service-cost stop, not a hard
dollar ceiling or a shared cross-arm pool. Validation requires the exact built-in
driver, jobs=1, zero infrastructure retries, and a persistent attempt journal.
Three arms have equal independent allocations; an in-flight operation can overshoot
before reporting. A hard total bill limit needs an external provider/account cap;
the harness neither configures nor attests that. Infrastructure is excluded and
opaque service fan-out is not counted as separately observed provider requests.

AttemptJournal.reported_spend reads ALL retained attempts, including failed,
unfinished and out-of-batch case IDs. No new table/schema/storage service. Unknown
final spend prevents new dispatch. The runner binds the remaining allowance to the
driver privately; it cannot exceed the pinned arm threshold and is never exposed
in model/backend case data. The loop checks charges between model requests,
batched tool calls and episode transitions. Exact-threshold final answers may be
delivered; above-threshold answers fail and disclose overshoot. Delivered final
answers with unknown cost remain answers but authorize no more work. An unknown
startup cost prevents the first model request. Existing no-policy behavior stays
compatible; this is an opt-in control, not a claim all arbitrary drivers enforce it.

Budget-stopped rows remain in the full cohort with explicit not_dispatched
metadata and zero-work preflight receipts in the existing journal. They do not
become missing tasks or free retries. Earlier costs survive resumed failures;
unfinished prior starts block further spend. One orchestrator owns a run directory.
Fresh output directories reset the local accounting boundary, never the real bill.

Added experiment/experiment.example.json, providers.example.json, prompt.txt and
README.md. Templates deliberately require operator-filled model/base URL, exact
Python/root, installed image IDs and numeric cost stop. No remote opt-in flag or
usable token is shipped. The common prompt/guidance is identical across arms.
Ordinary is real Python/StatsForecast, lean adds optional ledger+temporal MCP,
full retains its actual legacy profile. Extra service compute and legacy model
selection differences are explicit. Offline template identity validation succeeds
for all three arms; it does not prove Docker/model readiness.

33 new spending tests passed (session61359,6.73s); 49 spending/accounting checks
also passed (26942,4.75s). Early tests found a real lost-zero-preflight-receipt bug;
fixed by persisting those receipts. The other initial failure was a test fixture
missing the changed driver in driver_files; fixed. No unresolved failure.
Existing real ordinary/lean/full episode tests now run with a cost stop enabled.
144 focused spending/episode/accounting/loop/matched/driver checks passed each on
Python3.11 (89852,22.61s) and3.13 (45323,23.55s), with actual containers.
Ruff source/changed workflow/tests, compileall on3.11/3.12/3.13, and whitespace pass.
119 documentation/boundary/progress checks passed (0.35s before final handoff).

Read-only cohort audit: existing workflow context-* files contain22 cases whose
oracles encode legacy context_behavior fields such as required_argument,
publication_mode and recommended_scenario_id. Merely filtering out old stage/
publish/quote flags does NOT make them tool-neutral. smoke.jsonl has historical
host-compiled stages/receipts; its capacity extrapolation should not be treated
as independently observed future truth. agent_episodes.jsonl remains a three-case
protocol smoke only. TemporalBench tasks.py/scoring.py and locally available
frozen task data are the next sources to inspect for reusable accuracy evidence;
do not import its historical forced/harness-recovered agent path into matched mode.

Next: finish a representative, independently scored held-out task cohort using
the existing driver/backends, then freeze and run an actual model comparison once
model/spending authorization is supplied. Avoid another runner/framework. Live
Paracast still needs deployment base URL and token environment-variable NAME,
not the secret. No reply to those choices yet. Useful offline cohort work remains,
so do not mark blocked solely for unanswered credentials. Original PLAN.md and
100-point gates remain unchanged.

Shipped src/gnomon/package inputs and both installed images are unchanged. Reuse
iteration17 image IDs and iteration13 wheel listed below. No new dependency/image,
paid request, remote write, deployment, commit/push or material deletion. No skills
or subagents used. Preserve the user's existing scratch files.

Final integration verification: full production + historical suite with both real
container image opt-ins passed: 3516 passed,13 skipped,360.63s (session50960).
All iteration19 handles are terminal; exact-owner Docker listing is empty.
Final tracking/whitespace checks are run after this documentation update.
Earlier iteration snapshots follow; their pending items are historical.

Iteration 18 complete. Objective active; score 97/100. This was concrete offline
progress, not a blocked turn. The remaining checks are actual matched agent
evidence (2 points) and deployed Paracast verification (1 point); scripted controls earn neither.

Added a genuine agent episode path, not another host-compiled outcome stage.
Case.episode has 2–8 ordered phases; the initial reveal is empty, phase names are
unique and the final oracle matches the case oracle. Existing empty-episode corpus
hashes remain byte-compatible. matched.prepare requires the exact built-in shared
driver and zero retries; _invoke independently refuses episode retries before start.
Future reveal plans contain no oracles and remain private to the driver, never
part of initial model/backend input. The same client/transcript/backends and one
cumulative round/tool/token/time budget span the complete episode.

Each unchanged submit_answer is durably appended before the next reveal.
AttemptJournal schema 2 adds append-only checkpoints with chained content hashes
and cumulative observed usage; v1 upgrades preserve old attempt bytes. Crashes
retain commitments and positive resource lower bounds but do not invent complete
totals. Resume refuses replay after ANY prior start, including a start with no
recovered commit. Scoring validates ordered checkpoints/final-answer binding,
grades every phase, uses weakest-phase correctness and preserves earlier safety
disclosures/forbidden-claim failures. Missing or altered phases cannot succeed.

Operator reveal events may add safe named files without overwriting agent state.
Combined lean ledger can append supplied actual revisions after commitment, with
no agent outcome-write authority. The operator never computes the answer: agents
request ledger evaluation or calculate from persistent Python files and submit.
Events have separate cost/digest records and share the wall deadline. Failed or
partial reveals stop the task without replay. Journals/hashes are accidental
corruption guards, not malicious-operator signatures or independent billing audits.

New three-case agent_episodes.jsonl is only a synthetic smoke corpus: last-value
forecast and actual revision, approval-dependent decision, source/recorded-time
replay. Its numeric oracles are independently checked. It is not a broad forecast
quality study. CI's existing container job includes the episode tests.
21 focused episode tests pass with actual ordinary/lean/full containers: persistent
ordinary/full Python files, actual lean-ledger forecast before later actual
ingestion/scoring, real shared-command scripted HTTP, no early reveal, crash/no
retry, immutable predictions, unknown usage, failed persistence/environment event,
schema migration and corruption grading. No real paid model was called.

Current artifacts/core are unchanged from iteration 17: ordinary image
sha256:fa55aa0eb37dd59d1ee258ab37d3acab35507d4f28192f3c159f813c17dd65b8,
service image
sha256:7ffb55e2aa02c29d374cec15b96f1cb4a275a5a6d3ffaa806138b6f8d70a40c8.
Reuse earlier wheel/core fingerprint and private Docker config; no new package
dependency or image build. No subagents/skills, commits/pushes, deployments, paid
calls or material deletions this iteration. Preserve pre-existing scratch files.

Next work: complete the bounded experiment configuration/spending policy and a
suitable held-out ordinary/lean/full task cohort using the existing driver and
backends; do not grow another agent framework or claim scripted fixture uplift.
Actual model/comparison choice and spending limit remain unanswered. Live Paracast
needs deployment baseURL and token environment-variable NAME, not the secret.
Full acceptance criteria in PLAN.md remain unchanged. Useful offline work remains.

Verification, all sessions terminal:
- Full production + historical suite, actual container opt-ins: 3483 passed,
  13 skipped, 340.81s (session51928). No test containers remain.
- Python3.11.14: 111 focused episode/accounting/loop/matched/driver checks passed,
  14.81s (7669), with real containers.
- Python3.13.11: same111 passed,15.30s (88350), with real containers.
- Python3.12:21 episode checks passed,5.03s (44484), with real containers;
  final focused108 passed/3 container skips,11.89s (40930).
- Documentation/boundary/progress119 passed,.50s.
- Ruff source/changed workflow/tests, compileall workflow on3.11/3.12/3.13,
  and git diff --check passed.
- Earlier session27182:16 tests passed but a temporary unused import lint failure;
  fixed and covered by all final results above. No unresolved test failure.

Earlier iteration snapshots follow; their pending items are historical.

Iteration17 complete. Objective active; score97/100. Previous turn made concrete
progress, not blocked. All test/build handles terminal; no test containers remain.

Added service_backend.py and an isolated service Dockerfile. Driver factories
benchmarks.workflow.service_backend:lean/full combine identical ordinary Python
with an actual persistent MCP service in a second container. Options require
software_image, service_image and local docker_host. Optional execution_options
{ledger:bool,temporal:bool} applies only to lean; full refuses it. Default tool
counts7 (Python+execution6),21 (Python+legacy full20); lean with both switches10.
Legacy forecast schema/selection is NOT aliased to current direct inference;
this is a declared profile comparison, not only a tool-count causal experiment.

Both use the same SoftwareBackend-owned isolation/lifetime/cleanup machinery.
MCP initialization/initialized notification/discovery/calls use actual stdio CLI.
Nonblocking pipe IO is bounded1MiB/deadline, validates strict JSON and matching
request ID. Protocol/timeout failure destroys its service; tool-error results stay
repairable. Close always tries both containers, even if one cannot confirm cleanup.
Agent Python cannot import Gnomon or access the service filesystem/ledger/host.
Each container has its own fixed resource budget: extra service compute is
disclosed, NOT called equal total compute or free infrastructure.

Service startup compares the entire installed Gnomon package's non-bytecode file
contents to checkout, including non-Python resources, not just version/.py labels.
Verified packageSHA3ae175a89c162538431acd63343c0db5ed08251a8b088ae0f3a32366b92b1595.
Python/distribution versions must equal ordinary's except added gnomon-forecast.
Hashes/versions are supplied local provenance, not remote weight or third-party
source attestation. Software bootstrap now rejects even unregistered Gnomon modules.

Public available_at_cutoff.files is an explicit mapping of <=32 simple names to
UTF-8 text, copied identically into /tmp/data in both containers. Whole public
input stays bounded; traversal/absolute/invalid names fail before daemon access.
No host-file reads or implicit conversion of observations. These are independent
copies: agent-generated Python files do not automatically cross to the service.
No future-stage data is preloaded; single-stage-only driver restriction remains.

Lean ledger/temporal config is root-owned readonly, written only at startup.
Agent outcome writes remain disabled. Real MCP tests save forecast[7], retrieve
it, reject agent append_actual, allow an explicit synthetic test-operator append,
score MAE1 and recheck saved[7]. This is service-path evidence, not a completed
staged agent protocol or host-filled answer. Full retains its old tracking model.

Service image gnomon-bench-service:iteration17 retained locally, ID
sha256:7ffb55e2aa02c29d374cec15b96f1cb4a275a5a6d3ffaa806138b6f8d70a40c8;
configDigestda9ea84949a24ef1ec7fd5d0f3167f53557bfbc1280477ab5c8033a6964bd9f7.
Built against same locked software and actual iteration13 core wheel from
/tmp/gnomon-operator-build-41nNwV; named contexts transferred only lock/wheel inputs.
Core installed no-index/no-deps; pip check passed. DockerfileSHA256
e781826ae5ca3a586427ba73a01697965734fea1b4c94749c2145c5726106a98.
Ordinary image remains iteration16-final ID
sha256:fa55aa0eb37dd59d1ee258ab37d3acab35507d4f28192f3c159f813c17dd65b8.
No new core build claimed; package sources unchanged. Dedicated CI software job
now builds a fresh core wheel/service and checks both backends. No remote CI run.
Docs: benchmarks/workflow/service/README.md and linked MATCHED/software guides.

Verification:70 combined service/software/loop/driver checks pass on3.12 (72923,
22.86s),3.11 (12295,26.27s),3.13 (61268,26.41s), all with actual containers. Includes
default profile schema/discovery, current/legacy forecasts, persistent data refs,
error repair, stale package rejection, optional ledger/temporal and cleanup.
Final production2451 passed/7 skips (18647,86.15s); historical1011 passed/6 skips
(10193,265.47s), both image variables explicitly enabled. Combined3462 passed/13
skipped.119 boundary/docs/progress checks (22709,0.79s), Ruff source/changed files,
three-version benchmark compileall and whitespace pass. All handles terminal;
last Docker listing by exact backend ownership label was empty. Images retained.

Next meaningful work: staged agent protocol and suitable forecast/decision/temporal
corpus. Preserve client messages/backends through real save/reveal/score journeys;
future observations must not reach the agent before reveal. Freeze/check the
agent's submitted prediction/state before reveal; do not recover answers from
engine facts or reuse historical host-compiled stages. Explicitly decide/document
profile and compute differences in the experiment identity, then establish total
spending/retry limits and run the actual matched model experiment. Existing fake
controls/real software tests do not earn ablation2. Live Paracast1 still requires
deployment baseURL and token env-variable NAME (not secret). Model/spending limit
were asked previously and remain unanswered. Offline work remains: not blocked.
No paid model calls, commits, pushes or deployments; no user images/containers
were pruned or overwritten.

---

Iteration16 complete. Objective active; score remains97/100. Previous turn was concrete
progress, not blocked. Do not restart the audit or rebuild working artifacts.

Added benchmarks/workflow/software_backend.py: the driver-compatible factory
benchmarks.workflow.software_backend:make accepts options exactly {image: immutable
local sha256 image ID, docker_host: explicit unix socket}. Only a local daemon and
already installed image are accepted. It offers ordinary Python with actual
StatsForecast/NumPy/pandas/SciPy and standard library, not per-library Gnomon
adapters. No Gnomon distribution is installed in the ordinary image. Agent tool
choice/code/final answers remain untouched. New Python process per call; /tmp files
persist within the task. Normal Python errors return stderr/status and allow repair.

Container has no network/host mounts, readonly root, dropped capabilities,
no-new-privileges, memory1GiB/no extra swap, pids128, CPUs2, tmpfs128MiB. Agent code
runs UID/GID65534. Trusted UID0 PID1 has only a finite sleep then exits; the agent
cannot signal it. It independently ends descendant work if the host disappears.
Stopped residue can remain in create/start crash window; scheduling/daemon/kernel
limits remain. This is not VM isolation or a hostile-code security certification.
Timeout/output overflow destroys the owned container, not only docker exec CLI.
Cleanup resolves exact ID, validates random owner label and never prunes/pattern
deletes. Missing Docker socket/ambiguous cleanup is not reported as absence.
No host environment or Docker auth config is inherited. Inputs/public cases go
over stdin only; no oracle or stage/reveal data is preloaded. Saved files ephemeral.

bounded_agent now captures bounded optional backend_provenance, including actual
image ID, installed distribution/Python inventory, public-input SHA and limits.
Zero tool service charge excludes local infrastructure/energy; no free-compute or
billing-attestation claim. The software factory is only one matched arm component;
lean/full must retain equivalent software/data access and isolate Gnomon services.

software/requirements.in + generated hash lock target Linuxx86-64/Python3.12,
no source builds: StatsForecast2.1.1, NumPy2.5.2, pandas2.3.3, SciPy1.18.1;
26 locked distributions total. Dockerfile pins public python@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a
(actual Python3.12.14); remote manifest inspect confirmed the registry digest.
Small context only Dockerfile/lock, no source/cases/secrets. Build runs hash-enforced
wheel installation and pip check. Separate CI ordinary-software job explicitly
builds and executes Docker tests; default historical runs without opt-in skip them.
Docs at benchmarks/workflow/software/README.md cover use, provenance and boundaries.

Local image gnomon-bench-software:iteration16-final built successfully from final
recipe; actualID sha256:fa55aa0eb37dd59d1ee258ab37d3acab35507d4f28192f3c159f813c17dd65b8.
Initial image gnomon-bench-software:iteration16 also retained, ID
sha256:393b9d2fb2cab3c8999649f02d14bc8b3ab2abb56987542bf20b7090e40d978d.
Final build reused same runtime layers; only build metadata changed the index ID.
Runtime config digest7096ae4860cc6ad55298c2bcc029a9b31aefa4e44ea4fe5b305b975454d5185b.
Docker client/build metadata isolated under /tmp/gnomon-software-docker-px8lrj;
dependency-resolution cache /tmp/gnomon-software-uv-cache. Initial attempt using
read-only default Docker config failed before building, then explicit private
config succeeded. No user image/container was overwritten, pruned or deployed.

Tests actually execute SeasonalNaive/AutoARIMA/AutoETS forecasts, file persistence,
repair after Python errors, network/root-write/timer-signal/credential boundaries,
resource settings, timeout/output cleanup, owner mismatch and independent expiry.
Scripted model also uses this backend through the actual loop.12 software tests
including7 opt-in container tests;51 software/loop/driver checks pass on3.12
(53282,12.75s),3.11 (34831,14.05s),3.13 (25111,14.41s), all using final container ID.
Production2451 passed/7 skips (55705,88.91s). Source/changed-file Ruff and benchmark
compilation on all three pass. Historical992 passed/6 skips (80264,262.90s) with
container tests enabled; combined3443 passed/13 skipped.119 boundary/docs/progress
checks pass (8689,0.79s). All handles terminal; final ownership-label Docker listing
is empty (no test containers left). Core runtime/package inputs unchanged;
iteration13 artifacts stand. Dependency lockSHA256
4901404bccaed8d14c060514888620d70d30eafdbc8a073f5e2554b379e570dc;
DockerfileSHA2568270d7f00dd0cff31e304a379a5d53acbe41401fb5eaa76e5e8a9e69e75955d2.

Next: combine the same ordinary software with real current lean and explicitly
scoped full Gnomon service backends under tested container isolation/lifecycle.
Do not expose an unrestricted host Gnomon filesystem/import/ledger API to agents.
Then finish suitable task corpus and genuine staged agent-owned save/reveal/score
journeys, global experiment spending controls and actual matched model run. No
scripted fixture or old forced-tool adapter earns ablation2. Paracast deployment1
still needs baseURL and token environment-variable NAME, not its secret; actual
agent/model/spending limit remain unanswered. Useful offline work remains, so
goal is not blocked. No paid model calls, commits, pushes or deployment changes.

---

Iteration 15 complete. Objective active; score remains97/100. The preceding
turn made concrete progress. Do not restart the audit or inflate the score.

Added executable benchmarks/workflow/driver.py. It binds the common pinned prompt/
provider JSON bytes, model, generation and limits to run_agent; picks only the
declared arm's operator-owned module:factory/options and guidance; removes the
experiment/config paths from model-visible case data. It works as an absolute
script from arbitrary cwd by pinning checkout import roots. Actual backend
source/dependencies must be declared in driver_files; this is not comprehensive
attestation of undeclared dependencies, arbitrary operator code or remote weights.

Driver provider JSON schema1: llm{base_url,token_env}, backends{ordinary,lean,full},
each {factory:'installed.module:callable',options:{...}}. Factories receive public
case, selected options, a private temporary Path and timeout, and return the
existing tools/call/close backend. No Gnomon adapter per forecasting library.
Factories are trusted operator code, not permission for agent-generated host
Python. Normal cleanup removes temporary state; hard kills can leave temp dirs.

Named credential required, no ambient key/.env/default URL fallback. Numeric
loopback only without --allow-model-requests; remote additionally requires HTTPS.
No redirects or environment proxies. OpenRouterClient accepts optional explicit
request_opener while historical defaults remain unchanged. No paid calls made.
The flag is authorization, not a total dollar quota or remote cancellation.

Added process.py and routed Workflow external invocations through it. POSIX-only,
new owned process group; bounded stdin1MiB/stdout2MiB/stderr64KiB while reading;
SIGKILL group at timeout/output-limit and normal completion. Tests prove actual
descendant cleanup with a sleeping parent and a parent that already exits, both
pipe floods, closed-pipe timeout and input rejection before process start. This
does not kill deliberately escaped sessions, Docker workloads or remote requests
and is not a code sandbox/global memory limit. OS-uninterruptible work remains an
OS limitation; unsupported non-POSIX driver execution fails before dispatch.

14 local HTTP/real-command tests bind all three arm selections and run current
Gnomon inference through the journal/scorer. They deliberately use the same actual
backend in every arm plus markers, NOT a real surface comparison. No oracle,
provider-config paths or credentials reach model messages. Redirects/missing keys/
changed pinned config fail; uncertain spend remains unknown.7 independent process
tests pass.182 focused on3.12 (1672,10.14s);106 on3.11 (83068,11.68s) and3.13 (29685,
12.01s). Production2451 passed/7 skips (21809,87.45s).118 docs/progress checks pass;
source/changed-file Ruff, three-version benchmark compileall and whitespace pass.
Historical980 passed/6 skips (76764,245.40s); combined3431 passed/13 skipped.
All iteration15 handles are terminal. Final progress/whitespace checks pass.
Shipped runtime/package inputs unchanged; iteration13 artifacts remain applicable.

Next: actual strong ordinary-software backend and lean/explicit full backends, then
forecast/decision/temporal task corpus and genuine staged agent-owned ledger/reveal
journeys. Local .venv has no numpy/pandas/scipy/statsforecast distributions (checked
actual metadata); do not call it a strong baseline. A local Docker daemon exists,
but no new container/image/environment was created this iteration. A normal Python
tool in all arms with Gnomon additions is a candidate; it needs tested isolation
and a pinned, genuinely capable software image. Keep model/provider/data access
fair; do not use the old forced-tool adapter or host-filled answers. Remaining
controls: total paid experiment spending/retry budget and actual task rollout.
Paracast live1 still needs deployment baseURL and token env-variable NAME; actual
agent/model and spending limit also remain unanswered. Offline work still exists,
so this is not a blocked goal. No commits, pushes, paid calls or deployment changes.

---

Iteration 14 complete. Objective active; score remains97/100. This iteration made
concrete offline progress, not a blocked turn. Do not re-audit or restart the goal.

Added benchmarks/workflow/bounded_agent.py: one unsteered loop with operator-supplied
fresh client/backend factories. It discovers and records actual MCP tool specs,
uses auto tool choice, passes original arguments and accepts only agent-submitted
numbers. It does not force forecast tools or recover facts into answers. The real
current Python and execution/full MCP surfaces are exercised by scripted tests;
these are protocol tests, not agent-uplift evidence or the ordinary baseline.

Tool/round dispatch ceilings count malformed/unknown requests and reject mixed
submission/actions. Token counts are provider-reported after work; unknown usage
stops further work and measured overruns are disclosed, not called preempted spend.
An already-delivered valid final answer may survive unknown token usage. Factories,
backends and cleanup are trusted callbacks, not sandboxes; cleanup is timed and its
failure invalidates complete dollar accounting. Remote requests may outlive timeout.
The host runner still lacks descendant-process-group cancellation and bounded
captured output. Limits are per attempt, not a total multi-retry experiment quota.

OpenRouterClient.chat(single_attempt=True) is an explicit new opt-in: n must be1,
no sample-cache restore/write, no transport retry, no truncated-response escalation,
no missing-choice top-up. Strict duplicate/nonfinite/nonobject JSON refusal and
1MiB HTTP response reads apply only to this mode; historical default behavior is
retained. No endpoint/model/credential is selected or called by this change.

Budget-cap measurements now persist in append-only attempt receipts, including
host-observed subprocess timeouts, and normalized comparison rows. A later clean
attempt cannot erase an earlier cap; absent/old measurements remain unknown. The
receipt JSON addition is backward-readable without a SQLite schema migration.
benchmarks/workflow/MATCHED.md documents the factory protocol and these boundaries.

Verification so far:161 focused checks pass on3.12 (session2791,5.85s);110 focused
checks each on3.11 (57227,8.06s) and3.13 (63468,8.34s). Production2451 passed/7 skips
on3.12 (78836,83.25s).118 docs/progress checks pass (26159,1.04s). Source/changed-file
Ruff and three-version benchmark compileall pass. All those handles are terminal.
Historical959 passed/6 skips (16277,251.82s); combined3410 passed/13 skipped.
All iteration14 handles are terminal; final whitespace/progress checks pass.
Shipped src/gnomon and package inputs are unchanged this iteration; iteration13
installed artifacts remain applicable, but no new wheel/container build is claimed.

Next meaningful work: bind the loop to the shared pinned driver command/model/
prompt/config; provide strong ordinary software and current lean/explicit full
backends, actual task corpus and genuine staged agent-owned ledger/reveal journeys.
Do not relabel legacy forced-tool adapters or host-compiled outcome bookkeeping as
agent behavior. Establish outer cancellation/output and total spending boundaries
before a paid run. Keep the original ablation2 pending until reproducible actual
evidence exists. Actual Paracast deployment1 remains pending baseURL and token
environment-variable NAME (not the secret). Agent/model and spending limit were
also requested previously; no reply yet. Useful offline work remains, so this is
not a blocked goal. No paid calls, commits, pushes or deployments were performed.

---

Iteration 13 complete. Objective active. Verified score97/100: executable
onboarding/operations docs1 now verified. Remaining: surface ablation2 and actual
Paracast deployment1. The previous turn was concrete progress, not a blocked turn.

examples/provider_plugin is a separate installable Python package, not a core
adapter or dependency. Its stateless last-value callable and fresh-per-request
FittedMean factory illustrate the existing ForecastRequest/Result boundary. The
fitting object deliberately refuses reuse. providers.toml registers both installed
module entrypoints and a relative ledger with agent outcome writes disabled.
walkthrough.py run --output-dir NEW creates only a new directory, copies config/
request, and exercises three unique executions, original forecast[14,14], pending/
partial/complete/revised/source-replay/recorded-replay scores with MAE history
[null,1,1.5,4,1.5,1.5]. It records but never authorizes/executes a proposed decision,
backs up and restores to another file, and verifies exact preserved forecasts/scores.
All example data/decisions are synthetic; no real library/TSFM superiority claimed.

walkthrough.py backup --source SOURCE --destination NEW_DESTINATION uses SQLite
backup with read-only source and exclusive0600 destination; verifies application
ID, source quick_check, destination integrity/foreign keys. It does not initialize
or migrate the source ledger, overwrite destinations or follow destination symlinks.
Failure may leave a partial newly created destination; docs require validation/new
retry path. Independent tests include committed WAL data and backup of a v2 source
followed by migration of only the copy. No distributed/anti-admin audit guarantee.

OPERATIONS.md documents startup/OS permissions, temporal coordinates, timeout and
cost uncertainty, backup/candidate-only migration/rollback with post-backup write
reconciliation, and explicit legacy imports. The legacy-import Python block is
executed by a test with a real old registry; source bytes stay unchanged and unknown
history stays null. README/docs index/inference/offline/installation guides link
the executable example and distinguish current unreleased checkout changes from
potentially older remote releases. Offline MCP example now uses execution default;
explicit legacy core remains documented. No skill source was changed/needed.

scripts/offline_wheel_smoke.py adds optional --example-wheel, fresh installed
plugin Python/CLI/MCP/ledger/backup journeys and pip check. Child environments drop
PYTHONPATH/PYTHONHOME/GNOMON_MCP_PROFILE to avoid ambient checkout/profile contamination.
CI builds the provider example into example-dist separately and requires the
network-disabled journey. .dockerignore now permits only Dockerfile/package inputs
and excludes bytecode/.env files; no unrelated data/private agent state is sent.

Verification:131 focused example/docs/import/boundary checks pass. Final Linux
production matrix2451 passed/7 skips each: Python3.12,83.41s(session51872);
Python3.11.14,84.28s(session1800); Python3.13.11,81.60s(session69389).
Historical923 passed/6 skips/231.06s(session1218). Combined3374 passed/13 skipped.
All terminal; no source changes after these final source gates. Compileall src/
tests/examples/smoke passes on all three; source/example/script Ruff, ShellCheck
and whitespace pass;118 final docs/progress checks pass. All installer/build/
container/smoke handles are terminal.

Artifacts retained under /tmp (no global environment or user command replaced):
- /tmp/gnomon-operator-build-41nNwV/: final core wheel865109 bytes,SHA256
  db04d06a8dc67837c4bb01ef3bf0e6aff13c345cc4cdc2a37553243cb5e81780;
  sdist886076 bytes,SHA256521b634137465dd5d4dfdc6463693c220b5290159d08dbcefbf844e2e7e816d0.
- /tmp/gnomon-onboarding-0rcvi5/: example wheel1754 bytes,SHA256
  aef0491b69858eb2c530e3fa63359280bdb65fabb2fcf4ea435e74284098e4c9.
- /tmp/gnomon-operator-pkQi93/: exact documented local pip install of core+plugin,
  persistent walkthrough outputs and successful installed CLI inference/ledger query.
- /tmp/gnomon-install-check-cn7oSK/: bash install.sh --local with isolated release/
  command/cache directories; installed version and direct inference pass.
- /tmp/gnomon-uv-check-eN3fNc/: isolated uv tool install .; version/inference pass.
- Local Docker image gnomon-review:operator-pkqi93,ID
  sha256:efe67b5bdc4deebf0769e7bef3ae2469dda79b2a59e2dd0bff10a775f5e5cd26.
  Actual Dockerfile build passes with3.19MB permitted context. Image usergnomon;
  direct inference passes with no network, read-only root and /tmp tmpfs.
Core+plugin installed-wheel journeys pass on Python3.11/3.13(sessions14906/25809)
and local Docker python:3.12-slim --network none(session23208). Repo/wheel mounts
read-only; containers --rm. No publishing, commits, remote CI writes or paid calls.

NEXT: resume the actual ordinary/lean/full agent driver/task work detailed in
iteration12 and benchmarks/workflow/MATCHED.md. The controls alone do not earn
ablation2. Preserve a real ordinary-software baseline, explicit full/lean semantic
differences, actual inventory/usage receipts and shared enforced limits. Do not
substitute historical forced-tool/recovered-answer or host-compiled outcome stages.
The user's agent/model/spending choice and separate Paracast deployment URL/token
environment-variable name remain unsupplied. Useful offline implementation remains;
the goal is not genuinely blocked. No subagents or applicable skills this iteration.

Iteration 12 complete: matched-controls implementation verified. Objective active;
verified score remains96/100.
No ablation points awarded: the actual surface driver/task experiment is unfinished.

Added benchmarks/workflow/matched.py and MATCHED.md; run_workflow --experiment
reuses the external process protocol and attempt journal. The common JSON contract
pins one exact Python/script argv, declared driver dependencies, model/revision,
generation settings, prompt and provider configuration hashes, budgets and all
three deliberate arm descriptions. Concrete source/benchmark Python bytes, scored
corpus, Python binary/version and installed distribution versions enter the identity;
two '+dirty' labels cannot stand in for equal source. Unknown remote revisions stay
null. Public cases omit oracles and gain experiment settings; the host does not
choose a preferred tool or rewrite agent tool arguments in this mode.

Resume checks identity before dispatch. Post-run checks refuse changed inputs while
retaining raw observations/journal. Summaries seal observation and journal bytes;
the dedicated comparison CLI checks those artifacts, exact planned task rows and
equal controls before all-task completion-aware comparisons of all three pairs.
Failed tasks and unknown spend remain in the denominator. Historical promotion and
generic reports refuse matched experiments instead of bypassing these checks.
The normalized row projection is shared with the existing runner, not a new scorer.

Limits are explicit: requested model/configuration is not driver/weights attestation;
only process timeout is host enforced so far. Tool/token/round caps need driver
enforcement. Environment values, undeclared dependencies, edited installed packages
and remote drift are not comprehensively captured. No random execution order,
independent-task or malicious-operator protection is inferred. Staged historical
cases are refused because their host-compiled outcomes are not agent ledger work.
The test driver is SCRIPTED arithmetic, uses no Gnomon tools and is not advertised
as a current-surface ablation or LLM improvement.

Verification:121 focused workflow/accounting/matched/report checks pass.
26 independent matched checks also pass on Python3.11(4.70s)/3.13(5.33s), using
the temporary environments listed in iteration11. Production2440 passed/7 skips/
77.34s(session30630 terminal). Final historical923 passed/6 skips/255.14s
(session45953 terminal), including the historical-consumer bypass guard. Combined
unique production/historical suites:3363 passed/13 skipped. All test handles are
terminal; do not restart or poll old sessions. Only documentation/tracking edits
followed the final source gates.
Source and changed-benchmark Ruff, compileall benchmarks and whitespace pass;
118 final docs/progress checks pass. No shipped src/gnomon or package boundary
changes; iteration11 distribution/source-platform evidence remains applicable.
docs/installation.md now correctly uses --local for checkout/private-clone installs;
without it install.sh fetches GitHub main, not the edited checkout. Executable
onboarding/operating examples still remain a pending release gate.

Next required work: complete a shared unsteered agent driver, actual tool inventory
receipts, host/driver shared-budget enforcement and a suitable matched corpus for
forecast/decision/temporal work. Do not label an ordinary agent with no normal
software as a strong ordinary-software baseline. Define the full arm explicitly:
legacy forecast selection and direct inference have different computation semantics,
not merely different schema size. Driver-owned staged state/reveals are required
before measuring agent forecast retention/scoring; do not reuse compiled bookkeeping.
StdioMcpSession in benchmarks/cik/mcp_agent.py is an existing reusable transport,
but defaults its child to legacy full and a600s timeout: select profiles and bounds
explicitly. Existing agent_adapter.mcp forces _preferred_tool, compiles arguments,
and recovers engine answers; it is not the new autonomous driver.

An async question now asks which live agent/model and total spending limit the user
wants for the eventual comparison. Continue useful offline work without guessing
paid model calls. The separate live Paracast base URL/token-env name are still
unsupplied. Remaining gates: ablation2, operating/onboarding docs1, actual service1.
No skills/subagents, commits, remote writes, paid calls or deletes this iteration.

Iteration: 11 complete. Objective active. Verified score96/100.
Measurement metrics3 and release checks2 now verified; ablation2, executable
operating/onboarding docs1 and actual live service1 remain pending.

Added benchmarks/workflow/accounting.py, a small benchmark-only SQLite attempt
journal, separate from the forecasting ledger. Application ID GNAT, schema1;
UPDATE/DELETE refused. With a checkpoint path, observations.attempts.sqlite3 records
a committed start before invoking a process and an immutable completion afterward.
No checkpoint uses a session-memory journal; receipts remain in returned metadata.
Unknown interrupted starts survive resume. Retry/stage/resume receipts merge by ID;
original failures, error codes, usage and leakage measurements are not overwritten.
Historical aggregates retain observed lower bounds but do not attest lost usage or
prior safety. One orchestrator owns an output directory; internal --jobs is supported.

Observation resource scalars retain compatibility lower bounds. resource_fields
masks distinguish supplied measurements from default zeros through serialization;
old files without masks remain unverified. Optional cost_usd supported. Call/token
counts require integers; resources must be finite/nonnegative. resource_accounting
provides observed versus complete totals, unknown counts, failed/unfinished attempts
and leakage coverage. Completed clean retries cannot erase prior leaks or unknown
safety. Scoring refuses duplicates; cost gates require full call/token accounting;
leakage gate requires full measurement rather than just zero observed leaks.
Normalized gnomonbench output exports null unknown totals and explicit completion/
error status. Stage diagnostics cannot override host-measured initial usage.

Shared OpenRouter client now tracks absent/invalid provider usage and cache
completeness. Unknown billing stays null; observed cost is separate, tiny costs
are not rounded to free, overflow is disclosed, and old request-cache records with
no measurement mask stay incomplete. Current-process usage excludes restored cache
counters. Workflow agent normalization uses those new counters, preventing outer
retry journals from double charging provider cache history. If aggregate cache
history cannot be matched to outer receipts, totals remain incomplete, never free
or falsely fully accounted. No billing attestation, sandbox or internal model-call
count guarantee. Producer latency is not complete orchestrator wall time.

Verification:122 focused tests pass, including21 new independent workflow-accounting
checks, real subprocess timeout/parallel/CLI export, crash-before-checkpoint, retry,
stage/resume, unknown/legacy/cached usage and prior leakage propagation. Production
2437 passed/7 skips/80.74s (session47892 finished). Historical897 passed/6 skips/
234.99s (session93643 finished). Combined3334 passed/13 skipped. Source lint and
whitespace checks pass. No shipped src/gnomon changes this iteration; the existing
iteration10 wheel was revalidated with offline installed journeys on Python3.12,
3.11.14 and3.13.11 (sessions9512,32850,1941 finished).

Release checks added three independent ledger regressions: spawned processes
initialize/append/deduplicate without losing revisions; failed study inserts leave
no orphan payload/metadata; a fault-injected v2-to-v3 upgrade rolls back its DDL and
version marker, preserves existing executions, and can be retried. All10 ledger
tests pass. No production source fix was needed.

Final production matrix INCLUDING these tests: Python3.12,2440 passed/7 skips/
82.17s(session17528); Python3.11.14,2440 passed/7 skips/83.28s(session14827);
Python3.13.11,2440 passed/7 skips/79.82s(session34566). All finished. Combined
unique production plus historical suites:3337 passed/13 skips. compileall src/tests
passes on all three interpreters. Linux is the tested platform; no Windows/macOS
certification or real optional model-weight inference is implied.

Development environments (pytest9.1.1,PyYAML6.0.3,editable project):
/tmp/gnomon-platforms-acrt51/venv313 and /tmp/gnomon-platforms-acrt51/venv311.
Python3.13 executable: /tmp/gnomon-platforms-acrt51/python/cpython-3.13.11-linux-x86_64-gnu/bin/python3.13.
Cache/install roots all under /tmp/gnomon-platforms-acrt51; no user environment
variables, dependencies or global interpreters replaced. No subagents/skills used.
ShellCheck0.11.0 (shellcheck-py0.11.0.1) installed only into that temporary313env;
installer lint passes. Ruff src and git diff --check pass. All four runnable batch
configs validate with --dry-run; no datasets or model calls performed.

Fresh isolated sdist/wheel at /tmp/gnomon-release-checks-ublF4A/. Wheel865023 bytes,
SHA25665f0c274ca619df2ae178faba100953d8663da8c277b727d7edfd667bb656b84;
sdist885931 bytes,SHA2564310a11b6678c89a00670be7aec5a21f43d82b22d8d04115374cfb38f72c80a0.
Distribution size and packaged-skill boundaries pass. Final wheel installed/CLI/MCP
journeys pass in6.89s in local Docker python:3.12-slim with --network none, repo and
wheelhouse mounted read-only. Existing local daemon/image used, no service started;
container removed with --rm. No publications, commits, remote CI writes or paid calls.
All test/build/container sessions are terminal. Required CI job equivalents ran
locally; this is not a claim that GitHub Actions ran against unpublished changes.

Next: matched ordinary/lean/full measurement2; current old Workflow Bench profile
adapters remain legacy-only, not evidence of the six-tool default. Reuse public
case/arm-command and independent grader primitives, pin corpus/model/prompt/budgets/
provider and concrete code content identity, distinguish scripted fixtures from
actual LLM experiments. Current benchmarks.common.manifest only compares benchmark
and target; code_revision '+dirty' does not establish equal source contents. The
new ablation must fail on uncontrolled differences, not merely equal task IDs.
Do not alias old full forecast-selection semantics to the new direct-inference tool.
Remaining release docs1 and live1 remain pending until corresponding evidence is
audited. INFERENCE.md now describes schema3 and local-file concurrency accurately,
but a prose update alone does not earn the executable operating-docs gate.
Live Paracast URL/token-env name remain
unsupplied. Useful local work remains; goal is not blocked.

Iteration: 10 complete. Objective active. Verified score remains91/100.
This turn made concrete implementation progress, but measurement credit stays
unearned because a producer-side retry/resume accounting defect was confirmed.

agent_eval.py comparison schema0.2 now includes every task in delivered-success
rates and the paired McNemar test. Voided/noncompleted/error attempts cannot be
reported as delivered successes. Completion is measured explicitly or inferred
only from declared successful/submitted status; unsuccessful rows without evidence
remain unknown, with all-task lower/upper bounds. Submitted abstention and harness
noncompletion are distinct. Absent/null safety/error/budget/abstention fields are
unmeasured. Safety deltas use exact explicitly graded pairs and disclose their IDs.

All attempts in supplied rows contribute resource observations. Means disclose
coverage; observed totals are partial and full totals stay null until all rows are
measured. Nonfinite/negative/mistyped resources fail; overflow totals remain null
with disclosure. Conditional accuracy and matched-completed diagnostics are separate
from the headline. Optional success_probability produces exact target-labelled Brier
score and ten-bin reliability diagnostics; not calibration proof, not independently
verified as pre-outcome probabilities. Binomial test assumes independent task pairs
and no multiple-comparison adjustment. IDs do not attest task/prompt/grader identity.

Bounded JSONL loader:10k rows,1MiB physical row,128 nesting levels; duplicate keys,
task IDs, nonfinite metadata and bad types fail. _read_records/_decode_record are
shared with historical benchmarks.report, which now rejects malformed/duplicate
normalized records instead of silently dropping/overwriting them. Its shared-task
comparison uses compare_rows; unknown binary grades are disclosed, continuous
quality remains conditional and baseline imputation is hypothetical, not a bound.
Workflow Bench compare refuses incomplete/duplicate/mismatched case sets for
promotion, retaining raw all-case summaries for diagnostic use.

Verification:243 focused tests and69 final changed-source checks passed. Full
production2437 passed/7 skips/83.92s (session44070 finished). Historical871 passed/
6 skips/243.98s (session17568 finished). Combined3308 passed/13 skipped. Source lint
and whitespace checks pass. Final sdist/wheel and installed CLI schema0.2 cap/cost
comparison plus existing temporal/default/provider/ledger/result/legacy journeys
passed at /tmp/gnomon-metrics-build-Zkk1KS/ (session11761 finished). No shipped
source changes after those final gates. Only delivery/docs updates afterward.
All relevant sessions are terminal. Packaging reopened then reverified; score91.
No skill was applicable/used this turn. No subagents, commits, paid calls or deletes.

NEXT CONCRETE WORK (do not award measurement3 until fixed):
benchmarks/workflow/run_workflow.py _invoke only returns the final Observation with
attempt counts, losing earlier calls/tokens/latency. run_command drops prior failed
observations on resume and replaces them with fresh ones, losing their usage too.
A read-only mocked-provider diagnostic confirmed both: failed3 calls/100 tokens
plus success1 call/10 tokens reports only1 call/10 tokens rather than4/110. This is
verified current behavior, not speculation. Observation.from_dict also defaults
missing resource fields to0; timeout/subprocess errors create zero-valued resources
without knowing provider spend. Preserve append-only attempt receipts across retry,
workflow stages and checkpoint resume. Sum observed usage while explicitly keeping
unknown totals unknown; do not invent zero timeout costs. Keep raw historical
observations readable and unverified where lost costs cannot be reconstructed.
_run_one currently combines stage scalars but does not carry every followup's attempt
metadata; handle this boundary as well. Add failure/retry/resume/unknown-cost tests
and inspect downstream scoring/comparison gates before earning measurement points.

Then implement the matched ordinary/lean/full workflow2 using exact corpus/task,
driver/model/prompt/budget/provider identity and honest scripted-fixture limitations.
Reuse the existing external arm-command primitive where appropriate: public cases
exclude oracles. Current Workflow Bench agent_adapter only supports legacy core/
evidence/full; it is not a measurement of the new six-tool default. Avoid silently
aliasing contracts or treating old profile results as new evidence. The full legacy
forecast path has different selection semantics; keep actual arm differences explicit.

Remaining release docs1/platform+concurrency2/live1 unchanged. Python3.11.14 exists
at /root/.local/bin/python3.11;3.13 not on PATH. Actual Paracast deployment URL/token
environment-variable name remain unsupplied; no guessed paid calls. Useful local
work remains, so this is not blocked. Earlier entries below are historical snapshots.

Iteration: 9 complete. Objective active. Verified score91/100.

temporal_ops.py provides one lazily exported temporal_operation with five closed
operations: normalize, duration, shift, interval, order_events. Standard library
only; no provider calls, ledger writes, implicit clock or natural-language parser.
Normalize resolves explicit local zone/fold choices; gaps and ambiguous missing
folds fail. All instant arithmetic/comparison uses UTC. shift requires elapsed or
calendar mode; month-end clamping and target folds are explicit. Dates stay dates.
Duration returns exact decimal strings; nonempty half-open intervals cover Allen13;
bounded unique events retain stable, disclosed ties without causal inference.
Named-zone rules are host_zoneinfo_unversioned, not pinned/attested. Schema/input
bounds and missing-zone errors are explicit; no required dependency added.

Default session remains6 tools/8 with ledger, does not import temporal_ops.
Operator enable_temporal=true adds1 tool; tool arguments cannot enable it.
CLI temporal --arguments JSON/@file uses an empty provider session and full results.
Python temporal_operation is lazy. Large MCP event results use shared result refs.
README, CLI/quickstart/config docs and TEMPORAL.md reflect the actual contract.
Skill-creator instructions read completely; packaged guidance updated with only
the relevant optional-use distinctions. Skill validator passes.

Verification:288 initial focused tests and114 final temporal/session/receipt checks
pass.83 new independent temporal cases. Full production2399 passed/7 skips/83.23s
(session30184 finished). Final source lint and installed sdist/wheel smoke pass
(session21526 finished): /tmp/gnomon-temporal-build-VvV6qJ/. Installed temporal
Python/CLI/real MCP parity and default opt-out join the existing provider/ledger/
full-study/large-pointer/legacy journeys. No shipped source edits since these gates.
Historical compatibility868 passed/6 skipped/234.81s (session54932 finished).
Combined3267 passed/13 skipped. All relevant test/build sessions finished.
Progress checker confirms91/100, its7 tests pass, and whitespace checks pass.

Next iteration: measurement metrics3 and matched ordinary/lean/full workflow2,
then release docs1/platform+concurrency2/live1. Actual Paracast deployment URL and
token environment-variable name still not supplied; no guessed paid calls.
Read-only tracing confirmed agent_eval._summary excludes voided rows from costs;
compare_runs excludes the union of voided tasks from both success denominators and
assumes absent safety fields are false if either arm ever measured them. Fix all-task
completion/success/cost and paired success tests, keep answer-only diagnostics and
explicit measurement coverage, never impute unmeasured safety as success. load_runs
also needs finite/type validation and linear duplicate-ID detection. Historical
benchmarks/report.py repeats pairwise void exclusion; update shared reporting or
isolate its old semantics explicitly. Its baseline-score fallback is a hypothetical
imputation, not an observed lower bound; preserve conditional-quality disclosure.
Ordinary/lean/full fixtures must not claim paid LLM uplift from scripted runs.
Python3.11.14 exists at /root/.local/bin/python3.11;3.12 is active.3.13 not on PATH.
CI declarations alone are not platform evidence. No new subagents, commits, remote
writes or deletions this iteration. Earlier entries below are historical snapshots.

Iteration: 8 complete. Objective active. Verified score88/100.

Iteration8 implements result_refs.py: one bounded projection and gnomon_read
for exact canonical JSON text pages/JSON pointers. Small
payloads unchanged. Large responses have result_ref/partial/scalar summary, not a
truncated forecast pretending completeness. Default counts now6 tools/8 with ledger.
ResultLimits defaults:8192 compact UTF-8 payload bytes,16MiB individual result,
64MiB total temporary retention,16 receipts. Private session temporary files are
integrity checked, LRU-evicted and removed at close; no caller-selected paths.
Study IDs without a ledger reference this same bounded store (latest3 subject to
shared eviction); ledger study retrieval remains durable and independent.

session.call(...,compact=True) is the default shared tool projection; compact=False
and direct forecast/evaluate methods retain full programmatic data. CLI uses full
mode so it cannot return temporary references that expire at process exit. This
includes complete initial evaluation studies, not just dead study IDs. Oversized retention
errors disclose completed work and available execution/study IDs; do not retry
blindly. Limits do not sandbox provider execution or bound parser peak memory.

MCP now bounds physical input frames1MiB, drains oversize lines, rejects non-object,
malformed/nonfinite JSON, invalid UTF-8 and malformed IDs/params without killing
later requests. Actual stdin is read through its binary buffer; injected text
streams remain supported. It advertises only implemented protocol2025-06-18, not
arbitrary client text. Source checked against official lifecycle/transports specs;
link is in INFERENCE.md. Default tool errors are bounded through the same projection.
Full legacy schemas/projections and discovery are not covered by the new payload
byte budget; structured/text MCP duplication adds bounded wire overhead.

Skill-creator used this turn (main instructions read completely) to update packaged
guidance for partial results, targeted reads and avoiding duplicate inference after
retention failure. Skill validator passed. Current onboarding/counts/compatibility
and docs reflect6/8 tools. No arbitrary protocol/environment dependencies introduced.

Verification:65 initial focused checks pass; expanded72 pass after real
stdio invalid-UTF8/large-pointer and CLI-full-output checks.244 docs/tool/workflow
checks pass. Initial production run exposed missing repair options for new codes;
those now distinguish expired, corrupt, oversized and completed-but-unretained work.
Final production2316 passed/7 skips/67.00s (session59303 finished). Historical868
passed/6 skips/228.69s (session58421 finished); later changes to repair guidance and
full CLI study dispatch additionally passed26 error/reliability/recovery and56
session/result/evaluation focused checks. Combined regression count3184 passed/13
skips. Final build and installed smoke passed at /tmp/gnomon-results-build-1dNZwr/
(session61956 finished), including real large-result pointer retrieval and full
CLI study output. No relevant build/test sessions remain running. No production
source edits after the final production/build gates began. Ruff/diff/skill validator
pass. Result gate earns3 points and package is reverified; score88/100.

Next: optional bounded general temporal primitives3, unbiased completion-aware
metrics and matched ordinary/lean/full workflow5, remaining release docs/platform/
live checks4. Do not claim that a compact result ref is durable or that these byte
limits sandbox arbitrary providers. Remaining scope is unchanged. All earlier
iteration details below are historical snapshots.

Iteration 7 supersedes the historical default warnings below. Ordinary CLI/MCP
startup now uses the owned GnomonSession execution profile:5 tools by default,
7 with a ledger. Public serve() owns/closes one session and reuses frozen data
references across calls. Explicit core/evidence/decision/data/full profiles retain
advanced workflows. Duplicate describe and experimental mega profiles are retired.
Resolve startup profiles in dependency-free product_contract.resolve_mcp_profile;
do not recreate a second default-selection rule. Configured CLI capabilities now
uses the same startup TOML as inference/MCP, without importing legacy formatting.

Removed311 lines from active toolspec registration. Full pre-cull source is
archive/legacy/toolspec_pre_session.py; unregistered _run_unified/_run_track helpers
live in legacy_experiments.py, imported only by explicit diagnostics. Keep their
independent numeric tests; do not restore retired tool/profile registrations.
toolspec.visible_tools() can disclose default session schemas for compatibility,
but toolspec.runner_for() intentionally does not create stateful default sessions.
The actual default dispatch is mcp_server.serve(session=owned), not those helpers.

Updated README, quickstart, CLI reference, product claims and packaged use-gnomon
skill. Skill-creator guidance used progressive disclosure: default provider/session
instructions in SKILL.md, conditional legacy semantics in references/legacy-workflows.md.
Old quickstart retained under archive/legacy. Skill validator passed; installed
smoke explicitly checks the referenced skill file is packaged. No accuracy uplift,
SOTA weights identity, calibrated inference or automatic action claims added.

Historical benchmark callers now explicitly select legacy full instead of relying
on the product default. Serial direct calls use benchmarks.common.legacy_surface;
in-process CIK test sessions restore their selected environment, real subprocess
sessions explicitly select full unless another profile is requested. TemporalBench
and workflow launchers reject retired profiles. Old named result arms remain
readable; they are not measurements of the new default.

CI now separates pytest tests (production matrix3.11/3.12/3.13) from pytest
benchmarks/tests (historical adapter compatibility on3.12). No coverage deleted.
Four shipped batch configurations pass dry-run validation. Paid benchmark runs
remain separately gated. Platform declarations are not evidence that those
platforms have actually passed; the supported-platform release gate is still pending.

Final verification: production-only2302 passed/7 skipped/78.06s (session75513
finished); focused223 and final165 surface/docs/workflow checks pass; affected
historical pathways26 checks pass. Initial broad run13 failures/3156 passes/13
skips exposed implicit legacy selectors and obsolete skill/registry expectations;
all identified causes were corrected. Historical-only final suite868 passed/6 skips/
221.46s (session67548 finished). Combined independent coverage3170 passed/13 skipped.
Final119 docs/progress/CI checks pass. Final fresh build at
/tmp/gnomon-default-build-8dfa3q/ and offline installed smoke passed default plus
explicit core/provider/ledger journeys after ALL shipped source/docs/skill changes
(session34507 finished). No relevant build/test sessions remain running. No
production Python edits after the successful production-only run began. Boundaries
and suite separation earn five points; score85/100. Goal remains active.

Next work after finalizing iteration7: compact result/evidence byte limits and
full retrieval (surface.results3); bounded independent temporal primitives3;
completion/error/budget/calibration/cost measurements and matched ordinary/lean/full
workflow5; release docs/platform/live checks4. In particular default MCP still
needs malformed non-object message resilience and bounded framing/results; data
retention row counts do not bound parser peak memory or trusted provider execution.
Do not call current defaults production-complete on the strength of tool counts.

Current iteration supersedes the historical routing/conformance warnings below:
`study_routing.py` now implements explicit immutable-cohort recommendations with
independent source/recorded query clocks, per-fold history replay, version/lifecycle
checks, immutable execution matching and pretrained training-cutoff requirements.
It requires at least three matched folds and an explicit baseline; unavailable
evidence falls back without provider calls or action permission. Actual revisions
append a new immutable rescore with accurate zero-call usage, never overwrite the
original study. Plain files cannot claim historical revision availability or
silently bind naive timestamps to UTC. Requested and effective cutoffs are returned.

Both unsafe legacy priors are disabled: router.py no longer reads mutable model
leaderboards; adapter_promotion.route_shadow_adapter retains the explicit champion
because overwritten shadow outcomes have no recording-time vintages. Prior source
is recoverable under archive/legacy; no historical data deleted. Legacy structural
suggestions and statistical diagnostics remain explicitly non-authoritative.

Session/typed CLI/MCP share gnomon_route (ledger required). Execution discovery has
seven tools with a ledger, five without. The legacy default is STILL core/ten tools.
Conformance now distinguishes valid stochastic outputs from an explicitly required
deterministic replay and checks actual request mutation; diagnostic IDs do not count
as numeric differences. AdapterBench explicitly requests its deterministic gate.
Nullable session max_seconds schema now agrees with Python operator configuration.

Iteration 6 verification so far: focused193 routing/legacy/conformance checks passed;
after correcting the benchmark caller, focused31 adapter/routing checks passed.
Fresh sdist/wheel exists at /tmp/gnomon-routing-build-ZNbdZI/ and offline installed
smoke passed including timezone-aware study routing and immutable rescore retrieval.
Ruff and diff checks pass. Initial full run:3165 passed/13 skipped/one stale benchmark
failure, corrected. Final full pytest:3166 passed/13 skipped/288.13s (session75184
finished). Progress checker tests:7 passed. No production source edits after the
final full run started. No relevant test/build sessions remain running. Routing
earns three points; final verified score80/100. Revalidate packaging after the
next default/public-surface change.

Next iteration default-cull tracing: product_contract.DEFAULT_MCP_PROFILE is core;
CLI selects GnomonSession only for explicit execution/config. mcp_server.serve()
otherwise dispatches legacy toolspec. Change the real ordinary startup path, not
just a constant. Preserve explicit advanced compatibility. toolspec profiles
describe duplicates core; mega exposes run/track experiments excluded even by full.
Tests/conftest explicitly selects full for legacy regressions; default tests remove
that environment. Preserve this independent coverage while revising default tests.
Packaged smoke must verify both ordinary new defaults and explicit legacy journeys.
Product claims still describe a regulated-governance wedge; align claims/docs with
the scoped agent time-series product, without claiming forecasting or agent uplift.

Read PLAN.md and progress.json first. The user authorized implementation, recoverable culling, and sustained iterations until the scoped production gates are complete. Do not stop at a proposal. Do not award points for merely creating this plan.

Baseline revision: 0d3d72e. Pre-existing untracked root files must be preserved: `-`, `Continue`, `Current`, `Immediate`, `Use`, `accelerate`, `actual`, `cases.`, `optimizing`, `that`. No repository AGENTS.md or ancestor instructions were found. No subagents authorized.

Initial review: 64,261 Python source lines; 22 registered MCP tools, 10 default; a public `average` question returns latest (100 vs mean 4.5357). APIAdapter exists but discovery depends on local catalogue and ordinary MCP configuration is inconsistent. Forecast tracking overwrites scores and rereads original inputs at registration. TemporalStore preserves source vintages but needs local-recording-time replay.

Full baseline pytest completed: 2,991 passed, 13 skipped in 329.61s; source lint passes. Retired ReasoningBench moved to archive/benchmarks; active run_all rejects it; 34 cull-related tests pass. Exact scalar statistics corrected. A later full run exposed an observed measure=change regression, now explicitly abstaining when comparison windows are missing (12 focused regressions pass).

Implemented and exported: InferenceEngine, ForecastRequest/Result/Execution, AdapterCapabilities, ParacastProvider, TemporalLedger. User callables and fresh per-request factories execute independently of model selection; optional native batches and bounded explicit-version/deterministic caches. Requests/results validate timestamps/identity/units/uncertainty/capabilities. Legacy predict bridges reject covariates/panels they cannot forward. Package legacy exports are lazy; a fresh-process test proves inference avoids optional evaluation/context imports.

Paracast uses its real quantile wire schema, explicit/route/ensemble modes, configurable prefixed base URL, named covariates and deployment discovery independent of the local catalogue. Point forecasts are medians. It does not attest a weights revision; record unknown, do not fake one. Reject sample paths/multivariate outputs/fixed seasonal periods and quantiles beyond six significant digits. JSONTransport is shared by the old generic API adapter: bounded bytes/timeouts/GET retries, no POST replay, no redirects, environment auth and secret-safe errors. No live inference sent. An asynchronous question asks the user for base URL and token environment variable name (not the secret); no reply yet.

TemporalLedger is optional, separate SQLite schema/application ID v2 with append-only rows and transactional content-addressed payloads, unique executions, immutable actual revisions and pending/partial/complete/versioned evaluations. Schema 1 migrates transactionally to add imports. Matched comparisons and cutoff-aware decision/outcome replay are implemented. Explicit read-only legacy artifact/tracking imports preserve forecast outputs without fabricating history, revisions, original execution times or overwritten scores. Missing/unsealed inputs are labelled; incomplete imported requests cannot enter matched comparisons. Import ids are idempotent; source registries are untouched. Ledger calendar scoring requires timezone-aware times; naive legacy timestamps require an explicit UTC binding or remain unscorable. TemporalStore adds per-observation recorded_at, strict recorded replay, visible snapshot identity and actual content hashing; old migrated recording times remain unknown, never invented. Duplicate ingests preserve first-recorded times.

Iteration 3 adds GnomonSession (session.py) shared by Python, `gnomon infer`, `gnomon ledger`, and `gnomon mcp serve --profile execution --providers-config ...`. Operator TOML explicitly registers Paracast/callable/factory providers; no cwd search or agent-controlled URLs/auth/imports. Builtin baselines are available without configuration. Outcome/import writes require an operator startup boolean. Execution MCP exposes capabilities, forecast and optional ledger (three tools, ~3,617 schema chars with read/scoring operations). Legacy core remains the default and needs consolidation; do not equate this bounded execution profile with the entire final product. Both CLI and MCP direct inference avoid loading optional toolspec/runtime/evaluation/context/publication modules. MCP output-schema status enum was removed because it rejected legitimate legacy statuses such as valid.

New evaluated artifacts persist sealed history.json regardless of optional render settings. Tracking registration uses frozen observations for cutoffs/scales, including store/as-of and multi-target refused channels; no rereading original files. Scoring falls back to canonical artifact.json when forecast.csv is disabled. Existing legacy TrackingStore scoring remains a mutable compatibility view; use the ledger for revision-aware queries/routing rather than treating that old summary as immutable evidence.

Verification: final fresh full run passed 3,086 tests, 13 skipped in 300.30s (session 98215 finished). Five intermediate failures were useful docs/repair-option/parameter-classification/artifact-file coverage gates and are corrected with actual documentation/contracts. No numerical golden changes in iteration 3. 167 focused cross-surface/import/ledger/docs/runtime checks passed, and the final focused session/provider/import/progress run passed 83 tests. Ruff and git diff --check pass. No relevant test/build sessions remain running.

Packaging: iteration 3 isolated build produced sdist/wheel at /tmp/gnomon-session-build-29P4yW/. scripts/offline_wheel_smoke.py passed fresh no-index/no-deps installed legacy CLI/MCP and new provider-configured execution CLI/MCP/ledger journeys outside checkout. Script now checks history.json too. Package gate earns one point but must be reopened and rerun after subsequent surface/package changes. Supported-platform/full release-check points remain unearned.

docs/production/INFERENCE.md and docs/cli-reference.md document session/configuration, exact wire mapping, storage/migration semantics and limits. README links to this in-progress status; do not imply legacy default CLI/MCP paths already use the new registry.

User supplied GitHub repository: private TensorLink-AI/paracast is accessible through connector tools. Source commit f5d3f53b17e56d8c7ed0cbae61e043b8667f236a. See AUDIT.md for schema/route details. No remote writes. Deployment base URL may differ and is not yet verified. Full remote schema/app/service content was fetched through GitHub (can refetch at that commit).

Iteration 4 implementation: datasets.py now owns LoadedDataset/load_stage/load_stage_multi and their loading helpers; pipeline imports/reexports the identical objects. Recorded_as_of is available for persistent-store loading; files explicitly refuse to fabricate that history. Leakage lint now covers datasets.py and read_input_rows. DataReferences (data_refs.py, available as session.data) retains immutable-to-callers snapshots in an LRU bounded by max_data_refs=16 and max_data_rows=100000 (counting vintages, not just current values). Limits bound retention, not parser peak memory. References freeze content and never reopen mutated files/stores. Explicit series selection is required for panels. Exact mean/median/latest/min/max/sum summaries accept inclusive start/end datetimes and preserve declared unit labels; unknown fields and timezone/window contradictions fail. No implicit unit conversion or cross-series aggregation. Ref forecasts derive future grids and reject stale grids at/before as_of and target history valid after as_of. Covariates remain on the raw typed request path; no implicit covariate/reference join.

Session MCP now exposes capabilities, inspect, describe, forecast and optional ledger (five with ledger); legacy core default remains ten. CLI infer has mutually exclusive --input/--request; --input requires explicit horizon and defaults schema timestamp/value. File flags are forbidden with --request. infer deliberately bypasses legacy CLI remote-source/schema/horizon guessing so it uses the same loader/contract as Python/MCP and remains free of optional pipeline/evaluation/context imports. CLI response includes inspection provenance; refs die with the CLI process. Long-lived Python/MCP can reuse them. See docs/production/INFERENCE.md for actual examples and limits.

Iteration 4 semantics: legacy compiler rejects unknown fields (window/unit included), nonempty custom comparison/validation dictionaries, irrelevant method/period/differencing/explanatory settings, duplicate member weighting and ignored aggregation. Registered stationarity/decomposition/regression methods still reach their existing capability planner, preserving unsupported-method receipts rather than substituting methods. Explicit observed questions cannot inherit a parent forecast horizon; explicit contradictory horizons fail. Natural-language fallback now retains a predictive verb for explicit future-horizon cues, not a describe/horizon hybrid. Generic answer envelopes set automation_eligible=false and action_authorized=false; supported computations make no blanket calibration claim. Fitted numeric policy decisions retain qualification thresholds but explicitly state action_authorized=false and eligibility_basis=numeric_policy_thresholds_only. Intent version0.9, temporal-answer contract0.11.

Iteration 4 verification complete: fresh full pytest 3122 passed, 13 skipped, 301.55s (session20452 finished). Focused 131 passed, 3.00s (session68165 finished). First broad run's two integration failures (classification and added authorization field) were corrected with retained assertions. Build /tmp/gnomon-datarefs-build-YXJmLS/ produced wheel/sdist, and installed offline smoke passed (session83402 finished): new file inference/frozen summary and legacy/new provider/ledger journeys. Ruff source/scripts/touched tests and git diff --check pass. Scope and authority now earn six additional points. Tests added in test_data_refs.py; quantities/intent/parameter-authority tests expanded. Parameter audit now recursively covers execution-session union/request schemas, not just old toolspec top-level fields. No relevant tool/test/build sessions remain running.

Iteration 5 implementation: backtesting.py exports EvaluationBudget and evaluate_reference (lazy public exports). GnomonSession.evaluate returns full studies; gnomon_evaluate emits compact summaries or retrieves full evidence by study_id. CLI evaluate --arguments JSON/@file accepts a data inspection object plus the same evaluation options. Candidates exclude the explicitly named baseline; all are registered providers, never local catalogue guesses. Default budgets: max4 providers including baseline, max8 folds, max32 attempts; sessions add max30 dispatch-bound seconds. Operator evaluation_limits can change ceilings, tools may only tighten them. Python cancellation callbacks and KeyboardInterrupt retain completed work. No hard cancellation of trusted running Python is claimed. Remote fan-out/model-call counts remain unknown; every baseline/provider/error attempt counts and no model retries/fallback are hidden.

Matched point metrics (MAE/RMSE/bias) use only the intersection of successful folds across every provider, with all requested/planned/attempted/failed counts preserved. Reports include declared provider revisions/lifecycle and returned per-run metadata; unknown weights/training cutoffs remain unattested. Factory instances are fresh/closed each fold; cache reuse is bypassed. Snapshot.narrow freezes stricter source/recorded handles and cannot widen parent bounds. Recorded replay narrows BOTH clocks per origin. Repaired histories are refused except chronological reordering, because final repairs can encode later rows. Missing vintage history/gaps/capabilities yield explicit unscored folds. Inspection repair metadata is now deep-copied so a caller cannot mutate internal provenance to bypass this check. General covariate/reference joins and quantile calibration are not implemented by this point evaluator.

Ledger schema3 adds immutable content-addressed studies and study_executions foreign-key references, with transactional v0/v1/v2 migration. Study truth vintages remain inside study payloads, not silently imported into online actuals. Full evidence is retrieved by ledger.study(study_id, recorded_as_of=...) / ledger tool operation study. Sessions retain only latest3 reports without a ledger; tool summaries omit training vectors. Repeated studies get new IDs/cohort hashes; originals and executions stay unchanged. No new automatic provider selector has been claimed.

Iteration 5 verification: full pytest3150 passed/13 skipped/288.84s (session44813 finished); focused183 checks passed/4.39s (session85441 finished). Real local HTTP Paracast evaluation verifies attempts versus internal fan-out; live local MCP and CLI match Python study cohorts. Ruff and git diff --check pass. Fresh wheel/sdist /tmp/gnomon-evaluation-build-EM4zO5/ and offline installed smoke pass, including evaluate/study retrieval (session50250 finished). No relevant live tool/test/build sessions remain. Source unchanged since final full run/build. Five points earned for budget and matched evaluation; routing remains unearned.

Immediate next: iteration6 must consolidate the product. Implement cutoff-bound ledger routing, then switch/narrow defaults and recoverably remove redundant mega/describe experimental routes. Current execution profile is six tools with ledger (capabilities, inspect, describe, forecast, evaluate, ledger), but legacy core still defaults to ten. Preserve explicit legacy access for advanced workflows while making the ordinary path the default; do not declare this profile a finished product.

Semantics still to audit beyond these gates: optional properties may still accept measures whose exact interpretation is not executed (e.g. explicit period/residual_scale versus transition labels); do not silently substitute those. General date/event/interval primitives remain pending. Conformance_report still incorrectly treats stochastic second-call inequality as protocol nonconformance, and its default season2 is incompatible with Paracast's unsupported fixed-season input; separate repeatability claims from valid stochastic outputs.

Routing/measurement still open: all routing priors require source and recorded cutoffs on relevant immutable cohorts. Current router.py _tracking_prior reads mutable store.leaderboard/model_performance without ANY cutoff, then chooses fingerprint-weighted realized MASE. This cannot be reconstructed safely from overwritten legacy scores: disable/cull that authoritative prior and use the new ledger path, preserving old tracking summaries as explicitly historical/diagnostic data. Existing route still records tracking decisions; structural starting points are advisory and need no historical prior. Tests in test_router.py currently expect legacy priors to engage; replace with regression evidence that future/mutable summaries cannot choose a model, not an untested deletion.

Useful route implementation direction, not yet code: explicitly select immutable study IDs, enforce study recording cutoff and exact provider revision/lifecycle/task identity, validate historical fold inputs against the current frozen snapshot narrowed to each origin, and align outcomes at independent query source/recorded cutoffs. Revisions may require a new immutable rescore instead of trusting stale study truth. Do not implicitly import actuals or use unversioned remote weights as pinned evidence. Naive valid times need explicit semantics, not silent UTC binding. Retain no-prior/baseline fallback and distinguish recommendation from action permission.

Current agent_eval.compare_runs drops voided pairs and numeric summaries exclude their cost: add total completion/cost metrics without survivor bias, preserving conditional diagnostics where useful. Matched ordinary/lean/full ablation workflow and release suite selection remain. Live Paracast URL/auth and supported-platform checks still unverified. See progress.json for exact remaining weights.

Important evaluation trap found while tracing next work: repaired file values can encode later rows even if a final Snapshot assigns known_time=valid_time. New evaluation should reject affected repairs or reconstruct/re-repair each fold from original knowledge, not treat repaired final history as a prefix. Snapshot.series(cutoff=origin) applies source availability but retains the snapshot's final recorded_as_of; strict recorded replay needs an independently narrowed recorded boundary per fold. Current legacy pipeline's prefix fast path checks latest known times, not retained recorded times. Ledger evaluations table already stores append-only JSON per execution (schema2); online evaluate aligns ledger actual IDs. Plan any backtest persistence explicitly rather than fabricating ledger actual revisions from an unapproved implicit import.

The preceding trap is fixed in the NEW evaluator (Snapshot.narrow and repair refusal); legacy evaluated pipeline remains to audit/isolate. Further schema-consolidation edge: Python operator evaluation_limits can set max_seconds=None, while the tool schema currently only allows numeric max_seconds; align nullable semantics when consolidating schemas. Also consider bounding study/result bytes as well as counts; existing retention limits do not bound parser peak memory or a trusted callable's resource use.

No manual context-clearing tool is available. Persist updates here before context transitions. Never mark the goal complete or progress 100 while a release acceptance check remains unverified.

Final isolated-check note: running the progress test alone with the pytest executable exposed an import-path dependency masked by full-suite collection. It now loads the standalone checker by its exact repository file path; seven tests pass with both invocation styles. No production source changed after the final full run/build.
