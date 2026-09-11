# Accumulated-experience workflow benchmark

This benchmark tests the cost and correctness of **using accumulating evidence**.
See [current status and retained experiments](STATUS.md) before interpreting a run.
It is separate from the closed fixed-portfolio forecast-selection experiment in
`../ledger_optimization/CONFIRMATION_010.md`. Read [PROTOCOL.md](PROTOCOL.md) for
the objective, fairness contract, budgets, scope and confirmation criteria.

The initial objective is 20% fewer billed tokens per correctly completed evidence
checkpoint versus a capable persistent SQLite control, with completion and
forecast-quality noninferiority gates. Passing a scripted test or development
pilot does not achieve that objective.

From the repository root, using an environment with Gnomon's development source:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python -m benchmarks.experience_workflow.audit \
  --seeds 100 101 102 103 --rounds 24 --output results/experience-workflow/audit-NEW

PYTHONPATH="$PWD/src" .venv/bin/pytest benchmarks/tests/test_experience_workflow.py -q

# Live calls: uses ENGY_API_KEY or the existing local .env; never logs the key.
PYTHONPATH="$PWD/src" .venv/bin/python -m benchmarks.experience_workflow.agent \
  --seeds 100 101 102 103 --agent-seeds 7 --rounds 12 --workers 4 \
  --output results/experience-workflow/pilot-NEW

# Also works while the run is ongoing, using completed checkpoint artifacts.
PYTHONPATH="$PWD/src" .venv/bin/python -m benchmarks.experience_workflow.report \
  results/experience-workflow/pilot-NEW
```

Each output directory must be new; completed transcripts and decisions are never
overwritten. Progress summaries can be refreshed. Confirmation requires the
guarded freeze workflow below. Run both agent seeds and all 24 rounds for a full development iteration;
keep validation seeds 200–203 for the frozen development candidate. Do not tune
on validation repeatedly or reuse earlier Favorita confirmation as fresh evidence.
Create a `STOP` file in a live run directory to stop after the current paired
checkpoint finishes in each worker, preserving complete response/usage records.

After a candidate passes full validation on 200–203 and a current-source audit:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python -m benchmarks.experience_workflow.freeze \
  --validation-run results/experience-workflow/validation-NEW \
  --audit results/experience-workflow/audit-NEW/report.json \
  --output results/experience-workflow/freeze-NEW.json
PYTHONPATH="$PWD/src" .venv/bin/python -m benchmarks.experience_workflow.agent \
  --confirmation-freeze results/experience-workflow/freeze-NEW.json \
  --workers 4 --output results/experience-workflow/confirmation-NEW
```

The freeze command checks the exact validation grid, current code, complete usage,
completion/cost/quality thresholds and deterministic audit. The confirmation run
has 24 worlds × 24 rounds × two agent seeds × two arms (2,304 decisions). A global
cohort claim prevents rerunning the same reserved seeds under a new filename.
The runner will not open confirmation from a development progress chart alone. After all
confirmation gates pass, progress can set `objective_achieved=true`; development
always leaves it false. Forecast-superiority and real-data claims remain separate.

* `scenario.py`: seeded event stream and independent, plain-Python scoring oracle.
* `storage.py`: public Gnomon ledger adapter and indexed normalized SQLite control.
  Its complete reusable reference SQL is given to the control, not withheld.
* `audit.py`: parity, historical query immutability and deliberate-fault checks;
  reports no-memory and hindsight forecast performance separately.
* `agent.py`: same execution-bound completion tools, API and budgets for both arms;
  reset chats, persistent notes/queries, event receipts and full wire logs.
* `report.py`: all-attempt token accounting, paired world-cluster intervals,
  completion, quality, stage/family breakdowns and explicit pending gates.

Ingestion and historical candidate generation are shared automated services.
Their storage writes and elapsed time are reported, not counted as agent tool
calls. Query/model discovery and all API retries count toward agent work. Both
arms can access only arrived events through their own tools; neither has a shell
or a tool for private scoring truth. SQL cannot attach databases or write raw
events. The host supplies bounded correction feedback, never expected scores.

The agent must report both original and current evidence correctly, execute the
prescribed provider and select its actual execution ID. The host validates these
independently. Wrong facts, unexecuted selections and exhausted budgets remain
failures even if their fallback forecast happens to score well. Numerical evidence
correctness and forecast quality are separate columns.

Limitations: synthetic demand, automated shadow collection/ingestion, one fixed
evidence-use policy, one initial model, and development-only agents. A real-data
longitudinal replication and an independently frozen confirmation are still
required before claims about users' retail decisions or convincing ledger value.
