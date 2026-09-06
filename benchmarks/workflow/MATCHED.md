# Matched ordinary/lean/full evaluation

The former forced-tool `agent_adapter.py` has been removed. Its historical results
do not measure autonomous use of the current execution-default surface. The
retained driver preserves original agent choices, arguments and answers. Actual
matched model runs remain pending; harness tests alone establish no agent uplift.

`run_workflow --experiment PATH` now adds a matched-controls contract to the
existing external-command runner. It does not add another agent or model client.
The command receives the original public case, without its oracle, plus an
`experiment` object containing shared settings and the selected arm description.
No preferred tool or tool arguments are compiled in this mode.

## Contract

The JSON document has exactly these fields:

- `schema_version`: `1`.
- `evidence_kind`: `scripted` or `agent`; this is an operator declaration, not an
  attestation. Scripted arithmetic fixtures do not establish LLM improvement.
- `command`: one argv list shared by every arm, starting with the harness's exact
  Python executable and a driver script. The CLI `--arm-command` must match the
  resolved argv. Separate arbitrary executables are not treated as controlled arms.
- `driver_files`: the driver script and its local dependency files. Files must
  exist and be unique. External dependencies remain operator-owned.
- `common`: `model` (`id` and nullable `revision`), `generation` (explicit driver
  settings), `prompt_file`, `provider_config_file`, and `budget`.
- `budget`: inside `common`, declare `timeout_seconds`, `jobs`,
  `infrastructure_retries`, `max_tool_calls`, `max_rounds` and `max_tokens`.
  Timeout, jobs and retries must equal the corresponding runner CLI options.
  Optional `max_reported_cost_usd` is a positive per-arm reported-service-cost stop,
  including failed/resumed attempts, not a prepaid or provider-enforced dollar cap.
  It requires the built-in driver, one worker and zero retries. See the
  [operator-filled experiment templates](experiment/README.md) for configuration
  and the distinction between three per-arm allocations and a hard total limit.
- `arms`: exactly `ordinary`, `lean` and `full`. Each declares only `description`,
  `tool_contract` and `guidance`, as nonempty strings. These are the deliberate
  differences; model/provider/prompt/budget overrides cannot be hidden here.

Paths resolve relative to the experiment file. Use environment-variable references
for credentials in the provider configuration. Neither secrets nor credentials
belong in this public JSON document. The provider file's bytes are hashed, not
copied into reports; its absolute path is recorded. Provider settings and remote
weights are not independently attested. A missing model revision remains null.

The identity includes concrete source and benchmark Python bytes, driver files,
the full scored corpus, prompt/provider file hashes, Python binary/version and
installed distribution versions. It does not equate two `+dirty` Git markers.
This is an accidental-mismatch guard, not a signature against a malicious operator.
Environment values, undeclared external files, installed dependency source edits
and changing remote deployments are not comprehensively captured or sandboxed.

Run each arm with the same common experiment and command, separate output
directories, and its corresponding `--arm ordinary`, `--arm lean` or `--arm full`.
Do not change the input files between arms. There is no automatic paid experiment.
Resume validates the pinned identity before invoking the driver again. Inputs are
checked again after execution; a detected mid-run change preserves raw attempts
but prevents publication of a new matched summary.

Compare completed directories with:

```bash
python -m benchmarks.workflow.matched \
  --ordinary results/matched/ordinary \
  --lean results/matched/lean \
  --full results/matched/full
```

The comparator verifies that observation/journal files still match the summary,
then refuses different controls, source contents, corpora, duplicate/missing task
rows or inconsistent arm identities. Error rows must remain in the cohort. It uses
the shared completion-aware comparator for all three pairs; incomplete resource
measurements remain null. It does not equate matched requested settings with
verified driver compliance, infer random execution order or certify independence.
The POSIX runner starts an owned process group, bounds stdin to1MiB, stdout to2MiB
and stderr to64KiB, and kills that group on timeout, output-limit failure or normal
completion. This includes cooperative descendants that outlive their parent, but
not deliberately escaped sessions, containers or remote requests. Unsupported
platforms fail before dispatch. Captured-output limits are not a global sandbox
memory limit. Tool/token/round limits must also be enforced and reported by the
driver. One orchestrator owns each directory.
There is no separate promotion or publication-audit runner.

## Shared agent loop (building block, not an experiment)

`benchmarks.workflow.bounded_agent.run_agent` accepts a public case,
the common prompt and budget, supported generation settings, and two operator-owned
factories. It does not choose a model, endpoint, ordinary environment or full
surface for you. Supplying the factories explicitly authorizes their execution.

- `client_factory()` returns a fresh `OpenRouterClient`-compatible client with
  complete/unknown usage accounting. Each `chat()` makes exactly one transport
  attempt with a bounded, strictly parsed HTTP JSON response and an absolute
  deadline. There are no caches, sample top-ups, hidden retries or token-budget
  increases. Credentials and endpoint selection are explicit.
- `backend_factory()` returns a backend with `tools()`,
  `call(name, arguments, timeout=seconds)` and `close()`. `tools()` returns its
  actual MCP tool specifications, not names inferred from an arm label. `call()`
  returns `ToolReply(value, cost_usd=None)`. An optional `startup_cost_usd` follows
  the same convention: unknown by default; zero only when actually established.
- The loop supplies the discovered tools plus `submit_answer`, uses automatic
  model tool choice, and passes the original arguments. Only the model submits
  answer numbers; the host never repairs them from a tool result. It records the
  inventory and digest, rounds, attempted/dispatched calls and result digests.

Generation settings are limited to `temperature`, `reasoning_effort` and
`max_output_tokens`; unsupported controls are rejected, not silently ignored.
The caller must bind the declared model, prompt, generation and backend to the
matched identity. The loop alone does not enforce that binding or qualify an
arbitrary operator backend as a fair baseline.

Malformed and unknown tool requests consume the call budget. Batched requests
cannot exceed the dispatch ceiling, and submission cannot be mixed with actions.
Failed dispatch costs remain unknown unless reported. Reported dollar totals
cover LLM and tool service charges, not infrastructure or an independent bill.
Cleanup is timed and failed cleanup prevents claiming complete dollar accounting.

Tool/round limits are hard dispatch ceilings; cumulative tokens and elapsed time
are checked before/after work. Prompt tokens may exceed a limit before the provider
reports usage, and a timed-out remote operation may continue. These are disclosed
overruns, not preemptive spending guarantees. Unknown token usage stops additional
work, although an already-delivered valid final answer can remain usable. A backend
must honor its timeout; factories and cleanup are trusted callbacks, not sandboxed
or preempted threads. Message/inventory/transcript byte limits are not a global
memory limit. No leakage-free attestation is inferred from a successful answer.

Budget exits are exported as nullable `budget_exceeded` measurements. Attempt
receipts preserve a prior positive finding through recovery/resume; a later clean
answer cannot erase it. Old receipts or unknown earlier measurements keep the
aggregate unknown unless some attempt establishes a violation. These are
per-attempt cap observations. With `max_reported_cost_usd` configured, the runner
also checks every retained attempt in the arm journal, refuses unknown spend and
passes the remaining allowance to the shared loop. The loop checks service costs
between operations; unknown charges stop further work. An already delivered final
answer can remain usable with unknown cost. An operation can overshoot the threshold
before reporting; this policy does not guarantee a hard dollar ceiling. Provider
limits, infrastructure expense and opaque service fan-out remain external concerns.

`test_bounded_agent.py` uses scripted model responses, including calls to real
current Python/MCP Gnomon surfaces. It tests protocol behavior and unmodified
answers, not agent quality. There is no automatic paid call in this module. The
optional [ordinary Python/software backend](software/README.md) is available
separately and must be explicitly built and configured.

## Executable pinned driver

`benchmarks/workflow/driver.py` is the common executable binding for this loop.
Use its absolute path after the harness Python executable in the experiment's
`command`, and include it and operator backend source/dependencies in `driver_files`.
It also works when invoked outside the checkout working directory. No backend
module or forecasting library is imported merely by showing `--help`.

For this driver, `common.provider_config_file` is a strict JSON document with the
following shape (the module names below are placeholders for installed operator
code; the built-in ordinary and service factories are documented in the links below):

```json
{
  "schema_version": 1,
  "llm": {
    "base_url": "https://your-model-service.example/prefix/v1",
    "token_env": "MY_AGENT_MODEL_TOKEN"
  },
  "backends": {
    "ordinary": {"factory": "your_project.agent_tools:ordinary", "options": {}},
    "lean": {"factory": "your_project.agent_tools:lean", "options": {}},
    "full": {"factory": "your_project.agent_tools:full", "options": {}}
  }
}
```

Each factory is called as `factory(case=public_case, options=selected_options,
workspace=private_temporary_path, timeout=seconds)` and returns the backend described
above. It must isolate its task state and close owned processes/resources. The
temporary workspace is deleted after normal completion; hard kills can leave
temporary state behind. Operator backend code is trusted: it is not safe to run
untrusted Python there simply because a directory is temporary. Use a separate
tested code-execution isolation boundary for an ordinary coding-agent backend.
Options are explicit operator configuration; this is not an agent-controlled
module loader or a Gnomon adapter for each forecasting library.

The driver reads prompt/provider files only when their bounded bytes match the
runner's pinned SHA256. It binds the common model ID, generation and limits, adds
only the selected arm's declared guidance, and removes experiment/config paths
from model-visible case content. The selected backend receives public case data,
not the oracle or other arms' configuration. Inventory, experiment identity,
selected factory and nullable operator-declared model revision enter the report.
Unknown remote weights are not verified by repeating an operator model label.
Undeclared backend dependencies remain outside the pinning guarantee; declare the
actual source files used, and keep all configuration fixed across arms.

The named token environment variable is mandatory, including for local scripted
HTTP tests. There is no fallback to `.env`, ambient `OPENROUTER_API_KEY`, another
endpoint or a proxy. Redirects are refused. Without `--allow-model-requests` the
driver permits only numeric loopback addresses; `localhost` is deliberately not
resolved through ambient name service. Remote origins additionally require HTTPS
and the explicit flag in the shared experiment command. This flag authorizes
model calls; it is **not** a total dollar cap. A service may continue charging after
the local worker has timed out. Do not run a paid experiment without agreed model,
task, retry and spending controls.

`test_workflow_driver.py` invokes the real shared command against scripted local
HTTP and current Gnomon forecasts, through the actual journal/scorer. Its three-arm
fixture uses the same backend plus different binding markers by design: it proves
configuration selection, not surface uplift or a strong ordinary baseline.

## Still required before claiming a surface ablation

The independent `test_workflow_matched.py` fixture exercises controls/accounting.
That fixture intentionally uses no Gnomon tools and is labelled scripted. Run it with
`python -m pytest -q benchmarks/tests/test_workflow_matched.py`.

The ordinary software backend and its [isolated lean/full combinations](service/README.md)
are now executable and independently tested, with actual inventories and explicit
contract/compute differences. The remaining work includes a suitable matched
task set covering the intended forecast/decision/temporal use cases. The ordinary
arm must not be an artificially weak substitute for the software available to the
agent. Full enables optional ledger/temporal tools on the same execution contract;
old legacy-full results describe a different product and cannot be pooled.

An initial [11-task retrospective cohort](cases/MATCHED_RETROSPECTIVE.md) now
combines four frozen observational forecast windows with tool-neutral quantities,
decision/temporal tasks and committed episodes. Its forecast grader reports
continuous errors and missing-horizon coverage separately from all-task success.
It is small, historically exposed data, not certified uncontaminated holdout data.
The cohort's existence does not earn the actual agent-comparison gate.

Historical staged cases are currently refused in matched mode: their host-compiled
outcome bookkeeping is not evidence that an agent saved and scored a forecast.
New `episode` cases use the driver-owned commitment/reveal protocol below.

## Committed agent episodes

`cases/agent_episodes.jsonl` is a three-case synthetic smoke corpus: fixed-baseline
forecast revisions, approval-dependent decisions, and source-time versus
recorded-time replay. It checks workflow mechanics, not forecasting SOTA or broad
agent quality. Use a suitable held-out task cohort for the actual comparison.

Case schema v2 may declare `episode`. It contains 2–8
ordered phases, each with `name`, `revealed`, `answer_schema` and private `oracle`.
The first reveal must be empty; initial data belongs in `available_at_cutoff`.
The top-level oracle equals the final phase oracle. The runner requires the built-in
shared driver, a persistent attempt journal and zero infrastructure retries.

The model initially sees only the public case and first answer schema. The driver
receives the future reveal plan without any oracle; it does not expose this plan,
the journal binding or experiment configuration to the model or tool backend.
Each `submit_answer` durably appends the raw, unchanged submission before releasing
the next phase. One client, transcript, backend and cumulative round/tool/token/time
budget span every phase. The host does not calculate a replacement answer.

The journal upgrades additively to schema 2, retaining old attempt rows. Its
append-only checkpoints include a chained content digest and cumulative observed
usage. These are accidental-corruption guards, not signatures against a malicious
operator. A crash leaves the last commitment and positive usage lower bounds;
unknown final usage remains unknown. Resume never replays an episode after any
prior attempt start, even if no commitment was recovered. Use a fresh experiment
for a deliberately repeated trial; do not treat it as a free retry.

Optional backend `reveal` events run after commitment and before the next model
call. Software backends create supplied `revealed.files` without overwriting
existing files. With the lean ledger enabled, the operator may append supplied
`revealed.actuals` through the same validated ledger API. This imports data only:
the agent must request scoring and submit its own answer. No outcome-write
permission is granted to the agent. Environment events have separate cost/digest
records, consume the shared wall deadline and do not count as agent tool calls.
Failed or partially applied reveals terminate the task; they are not retried.

Scoring validates the checkpoint chain and final-answer binding, grades every
committed phase, preserves earlier disclosure/forbidden-claim failures, and uses
the weakest phase correctness. Missing phases or altered commitments cannot count
as completed successes. Container-backed tests verify saved Python files and actual
lean-ledger forecasts survive until delayed outcomes; scripted model replies in
these tests are protocol evidence only, not measured agent uplift.
These requirements remain in the production plan; control tests alone earn no
ablation acceptance points and support no LLM accuracy/cost-uplift claim.
