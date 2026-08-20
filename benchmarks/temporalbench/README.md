# TemporalBench — Contextual and Event-Informed Time Series Tasks

Adapter for [TemporalBench](https://arxiv.org/abs/2602.13272) (Weng,
Cao, Yang, Sharma, Liu — the USC Melady group). Official dataset:
[Melady/TemporalBench](https://huggingface.co/datasets/Melady/TemporalBench)
(Apache-2.0), with a public
[leaderboard](https://huggingface.co/spaces/Melady/TemporalBench_Leaderboard).

Four task families over retail / healthcare / energy / physical-systems
series: **T1** historical understanding (MCQ), **T2** context-free
forecasting, **T3** contextual reasoning (MCQ), **T4** event-informed
prediction — the closest public match to Gnomon's context-event contract.

## What is official, what is ours

Official (from the dataset, used as published):

- every task row of the labeled split
  (`task_merged_dev_with_labels_tiers.jsonl`): the complete prompts,
  inputs, labels, ground-truth futures, and MCQ options,
- **all forecast metrics**: the dataset ships
  `forecast_metrics_utils.py` (the authors' reference implementation,
  including the weighted OW metrics for multi-channel MIMIC rows); this
  adapter imports that file unmodified and scores every condition's
  forecasts with it.

Ours, and disclosed as such: **choice scoring is local**. ONLY the
forecast metrics go through the official module. T1/T3/MCQ answers are
graded by this adapter's own case-insensitive exact match against the
labels embedded in the rows (`scoring.py`); whether that is equivalent
to the official/leaderboard choice scoring is unverified. Treat choice
accuracies as this adapter's numbers — comparable across the conditions
here, not necessarily to the leaderboard.

Ours (this directory) — the three conditions:

| Condition | LLM | Numbers produced by |
| --- | --- | --- |
| `control` | official prompt verbatim via OpenRouter | the LLM |
| `gnomon-pure` (T2/T4) | none | Gnomon; MCQs answered `Uncertain`, or the `ABSTAIN` sentinel (scores wrong) where no such option exists |
| `gnomon-agent` | answers choice questions given Gnomon's evidence | Gnomon — forecast arrays in the final answer are Gnomon's, not editable by the model |
| `gnomon-mcp` (all tiers) | drives the real `gnomon mcp serve` tool surface itself | T2/T4, per channel: a Gnomon artifact used verbatim, or the model's own values labeled `model` — the route is recorded per channel. T1/T3: the model's own answers, with the same tool surface available |

With typed questions, the MCP arm records `canonical_mcq` (immutable engine
answer), `synthesized_mcq` (model proposal), and the final governed `mcq`.
Supported answers bind the final view. Weak answers remain canonical by default;
an override needs the adjudicator to validate the exact alternative from at
least two independently supported evidence kinds. A context quote alone is
provenance, not authority, and evidence weights are never presented as
probabilities.
`choice_contract` scores all three views and reports override help/harm, so model
reasoning is measurable without disguising it as an engine-authored answer.

Disclosed adapter decisions (see `gnomon_runner.py`): rows carry
index-aligned arrays rather than timestamped observations, so Gnomon models each
channel on a regular axis anchored to the row's official `history_end` and
`cluster_start` (index-based metrics — the axis never enters the score). This
preserves the calendar alignment of T4 events while making the unavoidable
regular-grid assumption explicit; nulls go through Gnomon's disclosed repair
layer; channels Gnomon abstains on stay absent and the row is recorded as
an abstention; `gnomon-pure` answers MCQs with the option sets' own
`Uncertain` — an honest abstention, reported as such. Questions whose
option set has no `Uncertain` are answered with the `ABSTAIN` sentinel,
which matches no real option and therefore deterministically scores
wrong (recorded as an abstention — never a guess that could luck into
the label).

`gnomon-mcp` (see `mcp_agent.py`) is the arm that measures how an
actual MCP agent uses Gnomon — the other Gnomon conditions run the
engine in the harness. Per row it writes the channels to one wide CSV
on the same synthetic hourly axis as every other condition, starts a
real `gnomon mcp serve` subprocess jailed to the run directory (the
session, verbatim tool-spec conversion, and path jail are reused from
`benchmarks/cik/mcp_agent.py`, per `docs/design/cik-mcp-tool-arm.md`),
and hands the model the official prompt verbatim plus every server
tool unpruned. `submit_answer` takes, per channel, exactly one of: an
`artifact_path` from a `gnomon_forecast` call in that run — used
byte-for-byte, and the artifact's own `target_column` must match the
channel, so a run cannot be mislabeled onto another channel — or the
model's own `values` (labeled `model`), or an abstention (explicit or
by omission).

For T2/T4, the adapter also compiles the official `future_covariates` into
Gnomon's point-in-time covariate channel. Rows are scoped per target and use
that target's observed-only axis, so different missing-value positions cannot
misalign a shared feature. `time_position_in_day` is declared as
`cyclic_1440`, yielding sine/cosine features rather than treating 23:59 and
00:00 as far apart. The runtime still admits it only after identical-fold,
leakage-safe ablation; summaries report channels considered and admitted.

Add `--compile-context` to measure the complete host integration rather
than numeric MCP execution alone. On T3/T4 the host runs Gnomon's owned
context-investigation prompt and schema first, excludes the large Input JSON
from the source document, verifies proposed quotes verbatim, and passes only
accepted events into `gnomon_forecast` or `gnomon_run`. The agent receives the
accepted/rejected receipt. T1/T2 do not pay a compiler call. Summary economics
report compiler calls and proposal counts separately from engine calls, and
report the numeric engine's later considered/admitted/rejected/applied counts
separately. Compiler acceptance proves that text was grounded; it does **not**
claim that the event was eligible to alter a forecast. Use
the same flag with `core`, `describe`, `evidence`, `mega`, and `full`: context
compilation is shared host infrastructure, so the experiment varies the tool
surface rather than whether text was connected to the product.

On T4, an artifact may contain a separately labelled
`hypothetical_sensitivity` path when the narrative grounds an event direction
but the short benchmark history cannot estimate its effect on four separated
folds. The adapter never submits that path as the forecast: the governed
history-only primary remains the sole headline score. It additionally computes
a retrospective diagnostic by overlaying scenarios only on channels where they
exist, and reports coverage, wins, losses, ties, and the sMAPE delta beside an
explicit `retrospective_overlay_never_submitted` warning. This measures whether
the standardized sensitivity carried information without turning hindsight
into a deployment selection policy.

```bash
for profile in core describe evidence mega full; do
  python -m benchmarks.temporalbench.run_temporalbench \
    --data-dir ~/temporalbench --condition gnomon-mcp \
    --mcp-profile "$profile" --compile-context \
    --context-receipts-dir results/tb-compiled-receipts \
    --model "$MODEL" --tiers T1,T2,T3,T4 --limit 40 \
    --output-dir "results/tb-$profile-compiled"
done
```

Run one surface first to populate the shared receipt directory; every later
surface verifies the task narrative fingerprint and replays the same receipt
without another compiler call. The summary exposes receipt reuse and counts
compiler calls separately. A changed narrative refuses the cached receipt
rather than silently compiling against different text.

Add `--compile-questions` to compile the T2/T4 choice-question text into the
same typed temporal-question contract used by the public tools. The compiler
sees question text and target names only—never options, labels, forecasts, or
future observations—and deterministic validation may accept or reject each
proposal independently. `--question-receipts-dir` persists immutable,
fingerprinted proposed/accepted/rejected receipts for matched runs. Summary
provenance reports question-compiler calls, receipt replays, accepted
questions, and rejected proposals separately from context compilation and MCP
engine calls. This arm tests host integration; the primary forecast remains
the same fitted executable and cannot be modified by compiler output.
Follow-up questions that omit a target inherit an explicit target from the
preceding question; explicit collective wording (for example, `all` or
`across`) remains aggregate. The raw model proposal is retained in the receipt
beside this deterministic discourse resolution.
`mcp_economics.choice_reasoning_stages` separates requested questions,
questions that reached a typed engine answer, official accuracy conditional on
that answer, and exact preservation of the canonical/display value by the host
agent. A compiler miss is therefore not reported as an estimator failure, and
an agent paraphrase is not reported as missing engine coverage.

For a model-supply experiment on histories too short for Gnomon's separated
selection/calibration/test contract, `gnomon-agent` and `gnomon-pure` accept
`--named-tsfm NAME`. This calls the pinned sandbox model directly and labels
every channel `experimental_named_model`; it does not claim that the model won
Gnomon's local evaluation and must not be reported as the governed default.
The agent returns choices only—the harness injects the immutable model arrays—
so output truncation cannot corrupt or duplicate forecasts.

```bash
python -m benchmarks.temporalbench.run_temporalbench \
  --data-dir ~/temporalbench --condition gnomon-mcp \
  --mcp-profile evidence --compile-questions \
  --question-receipts-dir results/tb-question-receipts \
  --model "$MODEL" --tiers T2,T4 --output-dir results/tb-compiled-questions
```

An artifact whose run abstained (`support:
"unsupported"`) is rejected at submission with the honest options
restated, including retrying the tool with `best_effort: true`: on
this arm the *model* decides whether to take the engine's labeled
fallback (which is why the harness `--best-effort` flag does not apply
here). A `gnomon_forecast` call takes every channel at once (`target_column`
accepts a comma list), and the resulting artifact is submittable for
each channel it covers — bound to that channel's own result, so a
batched run cannot hand every channel the first one's numbers. The
prompt names the run's jail directory and that batching exists: both
were measured as pure waste in the first sweep (a round per row spent
learning the jail by rejection, six calls per row doing what one does),
and neither sentence argues for using the engine.

Per-channel routes (`gnomon` / `informed-direct` / `direct` /
`abstain`) and support labels flow into `details/`, the GnomonBench
records, `summary.json`'s `forecast_channel_routes`, and
`score_per_channel.py`'s support mix.

**T1/T3 run the same session** with the same unpruned tool surface —
whether an agent that can interrogate a series answers descriptive
questions better is the measurement, and pruning the tools to fit the
tier would settle it by construction — but with a prompt that never
mentions horizons or channels and a `submit_answer` whose schema is the
tier's own answer shape (T1: the row's label fields; T3: one choice per
packed question, in order). A right answer cannot be lost to JSON
formatting, which is a property of the harness rather than of the
model's temporal reasoning.

**Caps end the run; they do not delete what it produced.** The tool
budget (4 calls) and the token budget (500k) return a typed
"budget spent, submit now" result instead of voiding the row, and a run
that reaches a cap or the round limit (10) without submitting gets one
final message offering `submit_answer` alone — a partial answer counts,
an uncollected one does not. Only when that produces nothing is the row
an abstention, with the cap named and never a silent fallback. Such a
row is marked `row_abstained`: it is reported in
`summary.json`'s `rows_voided_by_harness` and kept **out of the choice
denominators**, because an answer the harness never collected is not a
wrong answer (counting it as one is what reported a 0% tier score for a
sweep whose rows had all hit the token cap mid-answer).

Tool results pass through verbatim up to 16k characters. Past that the
bulk — long forecast arrays, evidence blocks — is shrunk to its first
and last few entries with the full count kept, under an explicit
`harness_truncated` marker, and if that is still too large whole bulk
blocks are dropped largest-first with each one named in
`harness_dropped`. (A six-channel 29-step MIMIC row returns ~33k
characters either way, and every round re-sends every earlier result, so
one unbounded batched forecast is paid for ten times over.) What is
never dropped: support states, warnings, abstention reasons, recovery
actions, error codes, and the `artifact_path` holding the complete
numbers — pruning descends past anything carrying a disclosure rather
than taking it, so a squeezed result still names every channel's
support label. The result also stays parseable JSON at every squeeze
level: cutting the serialized text would leave the model unable to read
the `artifact_path` it needs to submit. A budget is a reason to send
fewer numbers, never a reason to send numbers without their
disclosures.

`--best-effort` (`gnomon-pure`/`gnomon-agent` only, **default off**) passes
Gnomon's own best-effort flag through: a channel that would abstain
publishes the engine's disclosed naive fallback instead, labeled
`support: "best_effort"` and carrying Gnomon's NO RELIABLE FORECAST
warning. Those rows are **not** supported forecasts; every consumer
keeps the label — each details record and GnomonBench record carries
`channel_support`, `summary.json` reports
`forecast_channel_support_mix`, and `score_per_channel.py` prints the
mix beside the compared scores. The flag exists because the official
all-channels rule voids a record over one abstained channel (see
"Comparing arms" below), so best-effort coverage of sparse channels is
the only way to keep the *official headline number* populated for an
abstaining arm; it stays off by default because trading an abstention
for unsupported numbers must be an explicit, labeled choice, never the
silent one.

## Comparing arms

The official all-channels rule scores a multi-channel record only when
**every** ground-truth channel is forecast; one missing channel voids
the record (`metric_flag: missing_channel`). That is the leaderboard's
rule and each arm's `summary.json` keeps reporting that official number
— it is the headline and it must not disappear. But it cannot compare
an abstaining arm against one that never abstains: on the MIMIC split
Gnomon abstained on the sparse `temperature_c` channel in 44 of 48
records (MIMIC charts temperature every few hours, so its history is
far shorter than heart rate's), which voided 38 otherwise-complete
records and left exactly one record comparable across arms.

Cross-arm comparison therefore goes through the **per-channel path**:

```bash
python -m benchmarks.temporalbench.score_per_channel \
    --data-dir ~/temporalbench \
    --baseline results/tb-control --treatment results/tb-gnomon
```

Long provider runs can be sharded without changing row identity: add
`--offset N --limit 1` to run exactly the Nth filtered row in a fresh process.
This is the recovery path when one row or provider session dies; combine only
shards produced with the same model, endpoint, profile, task filters, and code
revision.

It scores, with the dataset's own metric module (nothing reimplemented),
the intersection of channels both arms forecast in each record, and
prints **coverage beside every figure**: how many records and channel
slots each number rests on, which channels either arm skipped (counted
and named, never dropped silently), and the support-label mix of the
compared channels. Quote a per-channel figure together with its
coverage or not at all — a subset mean without its n is meaningless.
The record-level `summary.json` coverage fields
(`forecast_channel_support_mix`, `forecast_channels_abstained`,
`forecast_rows_scored`) serve the same rule for the official number.

## Setup and run

```bash
pip install huggingface_hub numpy   # numpy is required by the official metrics
export OPENROUTER_API_KEY=sk-or-...

python -m benchmarks.temporalbench.run_temporalbench --download \
    --data-dir ~/temporalbench

# Control vs treatment on the forecasting tiers (start small):
python -m benchmarks.temporalbench.run_temporalbench \
    --data-dir ~/temporalbench --condition control \
    --model openai/gpt-4o --tiers T2,T4 --limit 50 \
    --output-dir results/tb-control

python -m benchmarks.temporalbench.run_temporalbench \
    --data-dir ~/temporalbench --condition gnomon-agent \
    --model openai/gpt-4o --tiers T2,T4 --limit 50 \
    --output-dir results/tb-gnomon

# The tool-use arm: the model drives the real MCP server itself
# (every tier — T1/T3 get the same surface with their own answer shape):
python -m benchmarks.temporalbench.run_temporalbench \
    --data-dir ~/temporalbench --condition gnomon-mcp \
    --model openai/gpt-4o --tiers T1,T2,T3,T4 --limit 50 \
    --output-dir results/tb-mcp

# Any OpenAI-compatible endpoint, for models OpenRouter does not host
# (--base-url, or $OPENROUTER_BASE_URL; the resolved endpoint is
# recorded in summary.json's llm_usage and in manifest.json):
python -m benchmarks.temporalbench.run_temporalbench \
    --data-dir ~/temporalbench --condition control \
    --base-url https://api.example.com/v1 --model some-model \
    --tiers T2,T4 --limit 50 --output-dir results/tb-control-custom

# The no-LLM floor (free):
python -m benchmarks.temporalbench.run_temporalbench \
    --data-dir ~/temporalbench --condition gnomon-pure \
    --tiers T2,T4 --limit 50 --output-dir results/tb-pure

gnomon eval compare --baseline results/tb-control/gnomonbench.jsonl \
                  --treatment results/tb-gnomon/gnomonbench.jsonl
```

`--datasets` filters by source (e.g. `--datasets MIMIC`). Outputs:
`summary.json` (per-tier choice accuracy and mean official forecast
metrics — both over scored rows only, as their `*_scored_only` names
say; abstained, errored and harness-voided rows are excluded and
counted separately, so compare arms via
`benchmarks/report.py`'s matched join), `details/` per row,
`gnomonbench.jsonl`, `manifest.json` (run provenance).

Long runs may be split with disjoint `--offset`/`--limit` ranges. Merge them
with `python -m benchmarks.temporalbench.merge_shards --target RUN SHARD...`,
then invoke the canonical target with `--resume`. Duplicate task ids must be
byte-equivalent or the merge refuses; the normal runner then recomputes one
summary over the complete matched set.
Use `--resume --retry-voided` after fixing a harness-cap defect: successful
rows replay, while only prior `row_abstained` cases execute again.

Notes: the benchmark is new (Feb 2026) and its README announces
human-annotated updates — re-download before comparing across dates.
The healthcare rows are *derived* from MIMIC-IV (no raw records), but
read the upstream data-use terms before publishing per-domain results.
The blind test split and leaderboard submission are out of scope here;
this adapter targets the labeled benchmark split the leaderboard
expects local metrics from.
