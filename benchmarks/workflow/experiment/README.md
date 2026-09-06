# Preparing the actual matched experiment

These are operator-filled templates for the existing shared driver, not a new
runner or permission to make paid requests. They deliberately do not run unchanged.
Copy the three files into a private experiment directory; rename
`experiment.example.json` to `experiment.json` and `providers.example.json` to
`providers.json`. Keep credentials out of these files and the repository.

Before running, choose and freeze:

1. The exact harness Python executable, checkout root, model ID and model service
   base URL. Record an immutable model revision only if actually supplied; otherwise
   leave it null. The model must support the driver's tool-calling and usage contract.
2. Actual installed ordinary/service image IDs, built from the documented locked
   [software](../software/README.md) and [service](../service/README.md) images.
   Enter their 64-character local image hashes, not mutable tags. Both images must
   contain the same ordinary dependencies; service package bytes must match checkout.
3. A per-arm reported-cost stop as a positive JSON number. All three arms use the
   same value. There are three independent equal allocations, not a shared pool.
   This is a stop-after-reporting threshold, **not a hard dollar ceiling**. An
   operation can overshoot before its usage arrives. A strict total spending limit
   requires an independently enforced provider/account limit, including any paid
   tool services. The harness does not configure or verify that external limit.
4. A task cohort and execution order, decided before inspecting results. The
   `cases/agent_episodes.jsonl` is only a synthetic protocol smoke corpus. The
   [small retrospective cohort](../cases/MATCHED_RETROSPECTIVE.md) supplies 11
   forecasting, decision and temporal tasks with checked oracles and explicit
   historical-data limitations; it is not certified unseen model-training data.
   Full enables optional ledger and temporal tools on the same execution contract;
   lean uses the six default tools. Record any startup-option overrides.
   The service arms also have a second container's compute allocation.

Store the model token in the named environment variable through your normal secret
mechanism. Do not put the secret in command arguments, templates or reports.
After approval to use that model service and its spending limit, add
`--allow-model-requests` to the common `command` array. Its absence intentionally
refuses a non-loopback endpoint. Freeze the files before running any arm.

The following validates the common identity without constructing a backend or
contacting the model. Run from the checkout with its harness Python. Set the two
paths to your filled configuration and chosen corpus:

```python
import json
import shlex
from pathlib import Path
from benchmarks.workflow.matched import ARMS, prepare
from benchmarks.workflow.schema import load_cases

experiment = Path("/absolute/path/to/experiment.json")
cases = load_cases("/absolute/path/to/cases.jsonl")
spec = json.loads(experiment.read_text())
command = shlex.join(spec["command"])
for arm in ARMS:
    identity = prepare(experiment, cases, command=command, arm=arm,
                       timeout=120, jobs=1, retries=0)
    print(arm, identity["experiment_id"])
```

This validates configuration identity, not endpoint credentials or Docker readiness.
Run each arm through `python -m benchmarks.workflow.run_workflow` with the same
`--cases`, `--experiment`, `--arm-command`, `--timeout 120`, `--jobs 1` and
`--infrastructure-retries 0`; only `--arm` and `--output-dir` differ. The arm command
must exactly match the printed/configured argv after shell quoting. Use three fresh
output directories and preserve them together. Do not edit source/configuration
between arms or reinterpret a failed task as missing data.

The reported-cost policy reads every retained attempt in each arm's existing
journal. Failed, resumed and unfinished work counts; unknown final cost stops
further dispatch. No retries or concurrent workers are allowed with this policy.
The driver receives only the remaining allocation. It checks charges between model
requests, tool calls and phase reveals. Already delivered final answers with unknown
cost can remain answers, but do not authorize more work. Rows stopped before
dispatch remain explicit zero-work failures in the cohort. Receipts distinguish
these preflight stops from external calls. A fresh output directory resets the
local accounting boundary; it does not reset or certify your provider bill.

Compare the three sealed output directories using `python -m
benchmarks.workflow.matched --ordinary PATH --lean PATH --full PATH`. Report all
tasks, errors, incomplete costs, cap violations and declared profile differences.
Passing scripted fixtures is not evidence of agent-quality improvement.
