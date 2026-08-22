# The integrated arm: Gnomon as real MCP tools the model may use or skip

Status: accepted. Supersedes the `gnomon-router` arm as the "agent
chooses" posture. This spec is committed before the arm's first scored
call; the implementation lives in `benchmarks/cik/mcp_agent.py` and runs
as `--method gnomon-mcp` in `benchmarks/cik/run_cik.py`.

## Why the router was not enough

`gnomon-router` makes one completion that returns `{"route": ...}`, a
regex scrapes it, and a Python `if` dispatches. The model never sees a
forecast, never sees an event rejected, and never gets a second turn.
Measured consequences: the first routing prompt was a constant function
(68/68 chose gnomon), and on the four matched task-seeds of the revised
run the router chose gnomon for correct-sounding reasons, lost 4/4 to
control, and had no way to notice. A one-shot blind commitment is not
tool use.

The routing decision does not need its own arm at all. A model holding
real tools makes it implicitly and observably: calling no Gnomon tool
**is** the direct route. This arm therefore folds routing into genuine
tool calling, and the route becomes something we *classify from the
transcript* instead of something we ask the model to declare.

## Shape

Per task-seed run:

1. The harness writes the task's numeric history to `history.csv` inside
   a fresh per-run jail directory and starts a real `gnomon mcp serve`
   subprocess with the jail as its working directory.
2. `tools/list` is fetched and every tool's `name`, `description`, and
   `inputSchema` are handed to the model **verbatim** as OpenAI-style
   function specs. Nothing is pruned or paraphrased: hiding a confusing
   tool would hide the confusion this arm exists to measure.
   (`outputSchema` is dropped — the chat-completions tool format has no
   slot for it; this is the one disclosed loss.)
3. One harness-local tool is appended: `submit_forecast`, the only way
   to end the run with an answer.
4. The model receives the same information the control receives — the
   full numeric history and the task's textual context — plus the jail
   path of `history.csv`, the column names, and the forecast window. It
   then runs a free tool loop: any Gnomon tool, any arguments, any
   number of rounds within the caps. Tool results reach the model as the
   server's `content` text **unedited**, including `isError` payloads
   with their typed codes and `repair_options`.
5. The run ends when the model calls `submit_forecast`, or a cap ends it
   as an abstention.

## The two honest exits

`submit_forecast` accepts exactly one of:

- `artifact_path` — a path previously returned by a successful
  `gnomon_forecast` call in this run. The submitted trajectory is read
  from the artifact on disk, byte-for-byte Gnomon's numbers; the model
  cannot edit a digit. Submitting a path the run never produced is a
  repairable error, not an answer.
- `quantiles` — the model's own per-step `{q10, q50, q90}`, exactly
  `horizon` entries. This is the direct exit: the model writes its own
  forecast, as the control does, and is scored on it.

Both exits are converted to score samples by the same
`samples_from_quantile_rows` sampler already used (and disclosed) by the
gnomon-agent arm, so neither exit gets a different distribution shape.
The sampler applies the same stratified marginals, extrapolated tails, and
lead-specific permutations to both exits. It remains a three-quantile
approximation, but neither exit receives the former clamped/comonotonic bias.

The earlier draft's rule "the model never writes a number" was written
for a mandatory-Gnomon posture and cannot survive a free choice: a model
allowed to decline the tool must be allowed to answer. What replaces it:
**the model never edits a Gnomon number** — a submitted artifact is used
verbatim or not at all — and every self-written number is labeled as
such by the route taxonomy.

## Route taxonomy (classified post-hoc, never self-declared)

- `gnomon` — submitted an `artifact_path`.
- `direct` — submitted `quantiles` having made zero MCP tool calls.
- `informed-direct` — submitted `quantiles` after making at least one
  MCP tool call. Disclosed, not hidden: a model that consulted Gnomon
  and then wrote its own numbers is one of the most informative
  behaviours this arm can surface.

## Caps — a breach is a disclosed abstention

| cap | value |
|---|---|
| assistant rounds | 10 |
| MCP tool calls | 24 |
| total LLM tokens (prompt + completion, per run) | 250,000 |
| wall clock per run | 600 s |

A breached cap raises `GnomonAbstained` with the cap named in the
reason (`cap:rounds`, `cap:tool_calls`, `cap:tokens`, `cap:wall_clock`),
so the run scores as an abstention under the official cap-and-impute
rule (imputed at RCRPS 5.0). It is never silently downgraded to a
direct forecast — a fallback would launder a harness failure into a
model answer. A run that ends with no submission inside the round cap
abstains the same way.

## Path jail

The model supplies `input` (and other path arguments) as free strings,
and the benchmark machine caches the CiK datasets — future windows
included. That is a live leakage route. Enforcement, before any call
reaches the server:

- Arguments with path semantics (`input`, `output_dir`,
  `context_events_file`, `covariates_file`, `actuals_file`,
  `store_path`, and kin) must resolve inside the jail directory.
- Any other string argument that resolves to an *existing* filesystem
  path outside the jail is rejected too (belt and suspenders; quoted
  spans containing `/` stay admissible because they don't exist on
  disk).

A violating call is answered by a harness-authored error payload
(`code: PATH_JAIL`, `authored_by: "harness"` so it can never be
mistaken for a server payload), the call never reaches the server, and
the model may repair and continue. The server subprocess also runs with
the jail as its working directory, so relative paths land inside it.

## Caching and replay

No result reuse across seeds or tasks: each run is a fresh conversation
and a fresh jail. The official CiK result cache keyed by `cache_name`
applies at the run level, as for every other arm.

## What this arm measures, and the discipline

Primary outputs — descriptive, no pre-registration required:

- route distribution and per-route score deltas on matched task-seeds;
- surface usability: how often tool calls errored, which codes, and the
  repair rate (an error followed by a corrected successful call of the
  same tool in the same run);
- cap-breach and no-submission rates.

Score discipline: any *claim* built on this arm's scores (e.g. "the
integrated arm beats mandatory Gnomon") enters the existing
registered-comparison rules — same pinned code revision across all
compared arms, abstention counts quoted beside every mean, artifacts
committed next to every number. The arm may run without a hypothesis;
the claim may not.

Dry-run gate before any full-grid spend: the unit suite (stubbed model,
real in-process server) plus a one-task smoke run with a cheap model,
reading the trace to confirm the loop, the jail, and both exits behave.
Same gate philosophy that caught the router's degenerate prompt.

## Comparison set

The scored comparison this arm completes: `control` (DirectPrompt),
`gnomon-agent` (mandatory Gnomon), `gnomon-mcp` (this arm) — all at one
pinned revision, full grid, official seeds. `gnomon-router` is retired
from the comparison. Its historical results preserve the lesson; the
superseded executable arm has been removed from the live benchmark harness.
