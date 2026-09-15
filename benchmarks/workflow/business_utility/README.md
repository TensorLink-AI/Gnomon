# Business-utility evaluation workbench

This extends the existing matched workflow runner. It does not execute Codex,
Hermes or another model through a second harness. The pre-registration commit is
`798ae907`; read [COMMON.md](COMMON.md), [Eval 1](EVAL1.md), [Eval 2](EVAL2.md),
[Eval 3](EVAL3.md) and the prospective [amendments](AMENDMENTS.md).

**Live results are pending.** The user requested a baseline on top of 1.2.0.
The pinned 1.2.0 wheel was subsequently found in remote evaluation assets, and
86 checks passed against its verified sources on Targon. See [remote preparation](TARGON.md)
for the deployment selection, image builds and remaining model-configuration gate.
The original local development checks used 1.1.9 and are not relabelled.

## What runs now

Generate a deterministic development corpus (all forecasts/reference calculations
are offline; this does not call an LLM):

```bash
PYTHONPATH=src:. .venv/bin/python -m benchmarks.workflow.business_utility.corpus \
  --output results/business-utility-development/corpus
PYTHONPATH=src:. .venv/bin/python -m benchmarks.workflow.business_utility.report \
  --corpus results/business-utility-development/corpus \
  --runs results/business-utility-development/runs \
  --output results/business-utility-development/reports
```

The generator refuses an existing destination. With no observations, reports say
**not measured**, preserve all planned tasks, and produce no numerical treatment
effect. They never reuse old Ditto/Hermes results. Development corpora generated
with 1.1.9 must be regenerated and validated against 1.2.0 before confirmation.

| Evaluation | Per arm | Primary comparison | What a bad business outcome means |
|---|---:|---|---|
| Leakage | 80 | lean vs ordinary | A capacity-plan signoff whose withheld error exceeds the declared tolerance |
| Repeatability | 128 | lean vs ordinary, 64 fixed-policy runs | Wrong fixed-policy number, with unauditable/missing handoffs separately visible |
| Threshold | 80 | full vs ordinary | Action recommendation disagrees with the declared distribution's cost-optimal action |

There are 864 planned agent tasks in total, not 864 independent real-world cases.
Repeats/variants cluster by source window. No field dispatches or customer losses
are observed by this experiment. Forecast quality superiority is not tested.

## Existing runner integration

`prepare.py` accepts explicit model endpoint, named credential environment variable,
per-arm reported-cost stop and immutable local image IDs. It generates configuration
for **../driver.py**, validates all nine identities through **../matched.py**, and
makes zero model requests. Without `--development-only` it refuses a runtime whose
declared version is not 1.2.0. That check is necessary, not sufficient: verify the
source ref and byte fingerprints as well. `--development-only` omits remote-call
authorization and marks the launch plan ineligible for confirmation.

```bash
PYTHONPATH=src:. .venv/bin/python -m benchmarks.workflow.business_utility.prepare --help
```

After locating 1.2.0, choosing the model/account and allocation, building the existing
locked ordinary/service images, and committing protocol amendments, corpus and
configuration, use exactly the existing CLI, once per evaluation and arm, following
the committed `launch-plan.json` order:

```text
<python> -m benchmarks.workflow.run_workflow
  --cases <prepared>/corpus/eval1.jsonl
  --experiment <prepared>/experiment.json
  --arm-command '<exact shell-quoted command array from experiment.json>'
  --arm ordinary
  --output-dir <runs>/eval1/ordinary
  --timeout 180 --jobs 1 --infrastructure-retries 0
```

Use the same pattern for lean/full and then eval2/eval3. These placeholders are
intentional: no endpoint, credentials or budget is silently selected. The reported
cost allocation is per arm **per evaluation** (nine allocations), not a hard account
billing cap. Remote requests need the existing driver's explicit authorization flag.
Native Codex-account transport is not implemented by this workflow driver; it
currently expects its documented tool-calling chat-completions-compatible client.
Reusing the old Ditto runner would violate the requested matched-runner constraint.

For a durable sequential launch after freezing the inputs, `dispatch.py` invokes
those same monitored commands in the committed order. Use
`python -m benchmarks.workflow.business_utility.dispatch --help`. It never retries
failed tasks and never substitutes another model or execution host.

`monitor.py` launches exactly that existing workflow command with one worker and
no retries, records CPU/available memory/owned-process RSS every two seconds, and
stops its owned process group at the registered memory thresholds. It requires
committed runtime/harness/prepared inputs and refuses the wrong release version.
Use `python -m benchmarks.workflow.business_utility.monitor --help` for its four
arguments. Before live execution, verify cleanup on a 1.2.0 smoke case.
The existing Docker backend caps each container at 1 GiB/2 CPUs with no swap;
service arms use two containers. This implementation has not run the live resource
guard or rebuilt/validated a 1.2.0 image. These are outstanding launch gates, not
claimed completed safety checks. No automatic continuation after a host incident.

## Implementation boundaries

`corpus.py` constructs public inputs and private grading data. All arms receive
identical source vintages in CSV and portable snapshot formats. Private holdouts
are only in the grader sidecar and are never passed to model/backend factories.
The existing runner also removes schema oracles before invoking the driver.

The shared loop has two opt-in, common-arm instrumentation flags: omit repeat IDs
from model-visible input, and retain at most 128 KiB/task of tool arguments/results.
IDs remain in journals. Receipts beyond the limit carry digests and an explicit
omission marker; absent evidence cannot prove the absence of leakage or repair.
No final answer is filled in or corrected from a tool result.

`backend.py` wraps the existing combined backend only for the full-arm threshold
reference. It adds an explicitly labelled overload to `gnomon_forecast`, not a new
tool or production package API. The existing service handles ordinary Gnomon calls
unchanged. Reference calls use identical supplied quantiles and their declared
continuous, piecewise-linear CDF. The wrapper is benchmark code outside the native
server; do not call those results shipped native MCP functionality.

`report.py` validates corpus hashes, and complete runs must pass the existing
matched comparator before receiving matched status. Counts include missing tasks;
numeric means disclose their smaller denominators. It exports source-stratified
cluster bootstrap intervals, cluster sign-flip tests and Holm adjustment only when
all three matched agent comparisons are present. It does not automatically award
an uplift claim; release, task-specific criteria and execution evidence still need
review. Self-reported repaired rows are explicitly not independently attested.

## Validation

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  benchmarks/tests/test_business_utility.py \
  benchmarks/tests/test_bounded_agent.py \
  benchmarks/tests/test_workflow_driver.py \
  benchmarks/tests/test_workflow_matched.py
```

Checks cover deterministic corpus bytes, identical repeat prompts, actual snapshot
cutoffs, fixed repair policy against the actual runtime, marginal arithmetic and
refusals, missing-answer accounting, unchanged agent answers, and matched driver
configuration. Scripted transport/property checks establish **no measured agent
uplift**, and they do not substitute for the requested 1.2.0 live replication.
